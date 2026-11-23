# Read-only DXGI adapter identity and LUID mapping. Creates no D3D device/resources.
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
public static class QwenDxgiAdapters {
    [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
    public struct AdapterDesc {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst=128)] public string Description;
        public uint VendorId, DeviceId, SubSysId, Revision;
        public UIntPtr DedicatedVideoMemory, DedicatedSystemMemory, SharedSystemMemory;
        public uint LuidLow; public int LuidHigh; public uint Flags;
    }
    public sealed class Row {
        public string name, counter_luid;
        public uint vendor_id, device_id, flags;
        public ulong dedicated_video_bytes, shared_system_capacity_bytes;
    }
    [DllImport("dxgi.dll", ExactSpelling=true)]
    static extern int CreateDXGIFactory1(ref Guid iid, out IntPtr factory);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    delegate int EnumAdapters(IntPtr self, uint index, out IntPtr adapter);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    delegate int GetDesc(IntPtr self, out AdapterDesc desc);
    static T Method<T>(IntPtr obj, int slot) where T:class {
        return (T)(object)Marshal.GetDelegateForFunctionPointer(
            Marshal.ReadIntPtr(Marshal.ReadIntPtr(obj), slot * IntPtr.Size), typeof(T));
    }
    public static Row[] Read() {
        if (IntPtr.Size != 8) throw new InvalidOperationException("Require x64");
        IntPtr factory;
        Guid iid = new Guid("770aae78-f26f-4dba-a829-253c83d1b387");
        Marshal.ThrowExceptionForHR(CreateDXGIFactory1(ref iid, out factory));
        var rows = new List<Row>();
        try {
            for (uint index=0; index<32; index++) {
                IntPtr adapter;
                int hr=Method<EnumAdapters>(factory,12)(factory,index,out adapter);
                if (unchecked((uint)hr)==0x887A0002) return rows.ToArray();
                Marshal.ThrowExceptionForHR(hr);
                try {
                    AdapterDesc desc;
                    Marshal.ThrowExceptionForHR(Method<GetDesc>(adapter,10)(adapter,out desc));
                    rows.Add(new Row { name=desc.Description, vendor_id=desc.VendorId,
                        device_id=desc.DeviceId, flags=desc.Flags,
                        dedicated_video_bytes=desc.DedicatedVideoMemory.ToUInt64(),
                        shared_system_capacity_bytes=desc.SharedSystemMemory.ToUInt64(),
                        counter_luid=String.Format("luid_0x{0:x8}_0x{1:x8}", unchecked((uint)desc.LuidHigh),desc.LuidLow) });
                } finally { Marshal.Release(adapter); }
            }
            throw new InvalidOperationException("Adapter enumeration exceeded bound");
        } finally { Marshal.Release(factory); }
    }
}
'@
$result = [ordered]@{
    recorded_at = (Get-Date).ToUniversalTime().ToString("o")
    scope = "Read-only DXGI enumeration and adapter memory counters; no D3D device/resources or settings changes. Capacity fields are not current usage. Counters do not certify physical residency/fallback policy."
    adapters = @([QwenDxgiAdapters]::Read())
}
try {
    $result.adapter_memory = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory |
        Select-Object Name, DedicatedUsage, SharedUsage, TotalCommitted)
} catch { $result.adapter_counter_error = $_.Exception.Message }
$result | ConvertTo-Json -Depth 8 -Compress
