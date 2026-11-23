# Owned, bounded D3D11 resources for PLAN.md's Windows application contention test.
# ABI slots checked against Microsoft's SDK: results/windows-d3d11-sdk-abi.json.
param(
    [ValidateRange(128,1024)][int]$MiB = 256,
    [ValidateRange(5,600)][int]$Seconds = 120,
    [string]$ReportPath,
    [string]$StopPath,
    [ValidateRange(0,2147483647)][int]$StopOwnedPid = 0,
    [switch]$ValidateOnly
)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
if ($StopOwnedPid -gt 0) {
    if (!$ReportPath) { throw "Require the owned process report path" }
    $owned = Get-CimInstance Win32_Process -Filter "ProcessId = $StopOwnedPid"
    if ($null -eq $owned) { '{"already_exited":true}'; exit 0 }
    if ($owned.Name -ne "powershell.exe" -or
        $owned.CommandLine.IndexOf($PSCommandPath, [StringComparison]::OrdinalIgnoreCase) -lt 0 -or
        $owned.CommandLine.IndexOf($ReportPath, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
        throw "Process identity does not match this fixture and its unique report"
    }
    Stop-Process -Id $StopOwnedPid -Force
    '{"owned_process_stopped":true}'
    exit 0
}
Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;

public sealed class QwenD3D11Occupancy : IDisposable {
    [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
    public struct AdapterDesc {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst=128)] public string Description;
        public uint VendorId, DeviceId, SubSysId, Revision;
        public UIntPtr DedicatedVideoMemory, DedicatedSystemMemory, SharedSystemMemory;
        public uint LuidLow; public int LuidHigh; public uint Flags;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct TextureDesc {
        public uint Width, Height, MipLevels, ArraySize, Format, SampleCount, SampleQuality;
        public uint Usage, BindFlags, CPUAccessFlags, MiscFlags;
    }
    [StructLayout(LayoutKind.Sequential)]
    struct Subresource { public IntPtr Data; public uint RowPitch, SlicePitch; }
    [StructLayout(LayoutKind.Sequential)]
    struct QueryDesc { public uint Query, MiscFlags; }

    [DllImport("dxgi.dll", ExactSpelling=true)]
    static extern int CreateDXGIFactory1(ref Guid iid, out IntPtr factory);
    [DllImport("d3d11.dll", ExactSpelling=true)]
    static extern int D3D11CreateDevice(IntPtr adapter, uint type, IntPtr software, uint flags,
        uint[] levels, uint count, uint sdk, out IntPtr device, out uint level, out IntPtr context);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate int EnumAdapters(IntPtr self, uint index, out IntPtr adapter);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate int GetDesc(IntPtr self, out AdapterDesc desc);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate int CreateTexture(IntPtr self, ref TextureDesc desc, ref Subresource data, out IntPtr texture);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate int CreateQuery(IntPtr self, ref QueryDesc desc, out IntPtr query);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate void CopyResource(IntPtr self, IntPtr destination, IntPtr source);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate void EndQuery(IntPtr self, IntPtr query);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate int GetData(IntPtr self, IntPtr query, IntPtr data, uint size, uint flags);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)] delegate void ContextVoid(IntPtr self);
    static T Method<T>(IntPtr obj, int slot) where T:class {
        return (T)(object)Marshal.GetDelegateForFunctionPointer(
            Marshal.ReadIntPtr(Marshal.ReadIntPtr(obj), slot * IntPtr.Size), typeof(T));
    }
    static void Check(int hr) { if (hr < 0) Marshal.ThrowExceptionForHR(hr); }
    IntPtr factory, adapter, device, context, query;
    readonly List<IntPtr> textures = new List<IntPtr>();
    public string AdapterName { get; private set; }
    public uint VendorId { get; private set; }
    public uint DeviceId { get; private set; }
    public long NominalResourceBytes { get; private set; }
    public int ResourceCount { get { return textures.Count; } }
    public int CompletedCopyPasses { get; private set; }
    public bool Released { get; private set; }

    public static QwenD3D11Occupancy Start(int mib) {
        if (IntPtr.Size != 8 || mib < 128 || mib > 1024 || mib % 64 != 0)
            throw new ArgumentException("Require x64 and128..1024MiB in64MiB steps");
        var session = new QwenD3D11Occupancy();
        try { session.Initialize(mib); return session; }
        catch { session.Dispose(); throw; }
    }
    void Initialize(int mib) {
        Guid iid = new Guid("770aae78-f26f-4dba-a829-253c83d1b387");
        Check(CreateDXGIFactory1(ref iid, out factory));
        for (uint index=0; ; index++) {
            IntPtr candidate;
            int hr = Method<EnumAdapters>(factory,12)(factory,index,out candidate);
            if (unchecked((uint)hr) == 0x887A0002) break;
            Check(hr);
            bool selected = false;
            try {
                AdapterDesc desc;
                Check(Method<GetDesc>(candidate,10)(candidate,out desc));
                if (desc.VendorId == 0x10DE && (desc.Flags & 2) == 0) {
                    adapter=candidate; selected=true; AdapterName=desc.Description;
                    VendorId=desc.VendorId; DeviceId=desc.DeviceId; break;
                }
            } finally { if (!selected) Marshal.Release(candidate); }
        }
        if (adapter == IntPtr.Zero) throw new InvalidOperationException("NVIDIA hardware adapter not found");
        uint level;
        Check(D3D11CreateDevice(adapter,0,IntPtr.Zero,0,new uint[]{0xB000},1,7,
            out device,out level,out context));
        QueryDesc qdesc = new QueryDesc(); // D3D11_QUERY_EVENT
        Check(Method<CreateQuery>(device,24)(device,ref qdesc,out query));
        var textureDesc = new TextureDesc { Width=4096, Height=4096, MipLevels=1, ArraySize=1,
            Format=28, SampleCount=1, BindFlags=8 }; // RGBA8, default usage
        // Incompressible initial pixels avoid treating a fast constant-color clear
        // as evidence of the requested physical allocation. Residency is still
        // measured separately; nominal resource bytes alone do not establish it.
        byte[] pixels = new byte[64*1024*1024];
        new Random(1701).NextBytes(pixels);
        var pinned = GCHandle.Alloc(pixels,GCHandleType.Pinned);
        try {
            var initial = new Subresource { Data=pinned.AddrOfPinnedObject(), RowPitch=16384, SlicePitch=64*1024*1024 };
            for (int i=0; i<mib/64; i++) {
                IntPtr texture;
                Check(Method<CreateTexture>(device,5)(device,ref textureDesc,ref initial,out texture));
                textures.Add(texture); NominalResourceBytes += pixels.Length;
            }
            Tick(); // Wait for actual GPU copies before reporting ready.
        } finally { pinned.Free(); }
    }
    public void Tick() {
        if (Released) throw new ObjectDisposedException("QwenD3D11Occupancy");
        var copy = Method<CopyResource>(context,47);
        for (int i=1; i<textures.Count; i++) copy(context,textures[i],textures[0]);
        Method<EndQuery>(context,28)(context,query);
        Method<ContextVoid>(context,111)(context);
        var get = Method<GetData>(context,29);
        var timer = Stopwatch.StartNew();
        while (true) {
            int hr=get(context,query,IntPtr.Zero,0,1);
            Check(hr);
            if (hr == 0) break;
            if (timer.Elapsed.TotalSeconds > 15) throw new TimeoutException("D3D11 copy fence did not complete");
            Thread.Sleep(2);
        }
        CompletedCopyPasses++;
    }
    public void Dispose() {
        if (Released) return;
        Released=true;
        for (int i=textures.Count-1; i>=0; i--) Marshal.Release(textures[i]);
        textures.Clear();
        if (query != IntPtr.Zero) { Marshal.Release(query); query=IntPtr.Zero; }
        if (context != IntPtr.Zero) {
            Method<ContextVoid>(context,110)(context);
            Method<ContextVoid>(context,111)(context);
            Marshal.Release(context); context=IntPtr.Zero;
        }
        if (device != IntPtr.Zero) { Marshal.Release(device); device=IntPtr.Zero; }
        if (adapter != IntPtr.Zero) { Marshal.Release(adapter); adapter=IntPtr.Zero; }
        if (factory != IntPtr.Zero) { Marshal.Release(factory); factory=IntPtr.Zero; }
    }
}
'@
if ($ValidateOnly) {
    $sizes = [ordered]@{
        pointer_bytes = [IntPtr]::Size
        adapter_desc_bytes = [Runtime.InteropServices.Marshal]::SizeOf([type][QwenD3D11Occupancy+AdapterDesc])
        texture_desc_bytes = [Runtime.InteropServices.Marshal]::SizeOf([type][QwenD3D11Occupancy+TextureDesc])
        gpu_api_called = $false
    }
    if ($sizes.pointer_bytes -ne 8 -or $sizes.adapter_desc_bytes -ne 312 -or $sizes.texture_desc_bytes -ne 44) {
        throw "Unexpected Windows SDK ABI structure size"
    }
    $sizes | ConvertTo-Json -Compress
    exit 0
}
if (!$ReportPath -or (Test-Path -LiteralPath $ReportPath) -or $MiB % 64 -ne 0) {
    throw "Require a new ReportPath and MiB in64MiB steps"
}
if ($StopPath -and (Test-Path -LiteralPath $StopPath)) { throw "StopPath already exists" }
$native = $null
$timer = [Diagnostics.Stopwatch]::StartNew()
$result = [ordered]@{ status="starting"; windows_pid=$PID; requested_mib=$MiB; requested_seconds=$Seconds;
    started_at=[DateTime]::UtcNow.ToString("o"); scope="Owned native D3D11 resource/copy consumer; nominal bytes do not certify physical residency" }
function Save-Result {
    $result.elapsed_s = $timer.Elapsed.TotalSeconds
    $temporary = $ReportPath + ".tmp"
    [IO.File]::WriteAllText($temporary, ($result | ConvertTo-Json -Depth 5), (New-Object Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $temporary -Destination $ReportPath -Force
}
try {
    $native = [QwenD3D11Occupancy]::Start($MiB)
    $result.adapter = $native.AdapterName
    $result.vendor_id = $native.VendorId
    $result.device_id = $native.DeviceId
    $result.nominal_resource_bytes = $native.NominalResourceBytes
    $result.resource_count = $native.ResourceCount
    $result.status = "active"
    $activeTimer = [Diagnostics.Stopwatch]::StartNew()
    do {
        $result.completed_copy_passes = $native.CompletedCopyPasses
        Save-Result
        if ($StopPath -and (Test-Path -LiteralPath $StopPath)) { break }
        Start-Sleep -Milliseconds 1000
        $native.Tick()
    } while ($activeTimer.Elapsed.TotalSeconds -lt $Seconds)
    $result.active_seconds = $activeTimer.Elapsed.TotalSeconds
    $result.status = "completed"
} catch {
    $result.status = "failed"
    $result.error = $_.Exception.Message
    throw
} finally {
    if ($null -ne $native) {
        $native.Dispose()
        $result.resources_released = $native.Released
        $result.completed_copy_passes = $native.CompletedCopyPasses
    }
    $result.finished_at = [DateTime]::UtcNow.ToString("o")
    Save-Result
}
$result | ConvertTo-Json -Depth 5 -Compress
