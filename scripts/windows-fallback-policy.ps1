# Default is read-only; -PreferNoFallback applies PLAN.md's explicit policy.
# Layout/function IDs come from NVIDIA's public SDK;
# setting ID/value names come from Profile Inspector's primary definitions.
param([switch]$PreferNoFallback, [switch]$EnumeratePolicyNames, [switch]$ValidatePrivateSetting, [switch]$ValidateSetterControls)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Collections.Generic;
public static class CudaFallbackPolicyReader {
    [DllImport("C:\\Windows\\System32\\nvapi64.dll", EntryPoint="nvapi_QueryInterface", CallingConvention=CallingConvention.Cdecl)]
    private static extern IntPtr Query(uint id);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int Initialize();
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int Create(out IntPtr session);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int SessionCall(IntPtr session);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int ProfileCall(IntPtr session, out IntPtr profile);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int GetSetting(IntPtr session, IntPtr profile, uint id, ref Setting value);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int SetSetting(IntPtr session, IntPtr profile, ref Setting value);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int SettingName(uint id, [Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder name);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int SettingIds([Out, MarshalAs(UnmanagedType.LPArray, SizeParamIndex=1)] uint[] ids, ref uint count);
    [StructLayout(LayoutKind.Sequential, Pack=4, CharSet=CharSet.Unicode)]
    private struct Setting {
        public uint version;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst=2048)] public string name;
        public uint id, type, location, isCurrentPredefined, isPredefinedValid;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=4100)] public byte[] predefined;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=4100)] public byte[] current;
    }
    private static T Function<T>(uint id) where T : class {
        IntPtr pointer = Query(id);
        if (pointer == IntPtr.Zero) throw new Exception("Missing NVAPI function 0x" + id.ToString("X"));
        return (T)(object)Marshal.GetDelegateForFunctionPointer(pointer, typeof(T));
    }
    private static void Check(int status, string call) {
        if (status != 0) throw new Exception(call + " returned NVAPI status " + status);
    }
    public static object[] Names() {
        Check(Function<Initialize>(0x0150e828)(), "Initialize");
        uint[] ids = new uint[] { 0x10ececc9, 0x1057eb71 };
        object[] result = new object[ids.Length];
        for (int i = 0; i < ids.Length; i++) {
            StringBuilder name = new StringBuilder(2048);
            int status = Function<SettingName>(0xd61cbe6e)(ids[i], name);
            result[i] = new { setting_id="0x" + ids[i].ToString("X8"), status=status, name=status == 0 ? name.ToString() : null };
        }
        return result;
    }
    public static object PolicyNames() {
        Check(Function<Initialize>(0x0150e828)(), "Initialize");
        uint count = 8192;
        uint[] ids = new uint[count];
        Check(Function<SettingIds>(0xf020614a)(ids, ref count), "EnumAvailableSettingIds");
        if (count > ids.Length) throw new Exception("Setting enumeration exceeded allocated array");
        List<object> matches = new List<object>();
        int resolved = 0;
        bool fallbackRegistered = false;
        for (int i = 0; i < count; i++) {
            fallbackRegistered |= ids[i] == 0x10ececc9;
            StringBuilder name = new StringBuilder(2048);
            int status = Function<SettingName>(0xd61cbe6e)(ids[i], name);
            if (status != 0) continue;
            resolved++;
            string text = name.ToString();
            if (text.IndexOf("fallback", StringComparison.OrdinalIgnoreCase) >= 0 ||
                text.IndexOf("sysmem", StringComparison.OrdinalIgnoreCase) >= 0 ||
                text.IndexOf("cuda", StringComparison.OrdinalIgnoreCase) >= 0)
                matches.Add(new { setting_id="0x" + ids[i].ToString("X8"), name=text });
        }
        return new { available_id_count=count, resolved_name_count=resolved,
            fallback_id_registered=fallbackRegistered, policy_name_matches=matches };
    }
    public static object Read(bool baseProfile) {
        int size = Marshal.SizeOf(typeof(Setting));
        if (size != 12320) throw new Exception("Unexpected NVDRS_SETTING_V1 size: " + size);
        Check(Function<Initialize>(0x0150e828)(), "Initialize");
        IntPtr session;
        Check(Function<Create>(0x0694d52e)(out session), "CreateSession");
        try {
            Check(Function<SessionCall>(0x375dbd6b)(session), "LoadSettings");
            IntPtr profile;
            Check(Function<ProfileCall>(baseProfile ? 0xda8466a0u : 0x617bff9fu)(session, out profile), "GetProfile");
            Setting setting = new Setting { version=(uint)(size | (1 << 16)), name="",
                predefined=new byte[4100], current=new byte[4100] };
            int status = Function<GetSetting>(0x73bf8338)(session, profile, 0x10ececc9, ref setting);
            if (status != 0) return new { profile=baseProfile ? "base" : "current_global", status=status };
            if (setting.type != 0 || setting.id != 0x10ececc9) throw new Exception("Unexpected setting ID/type");
            uint current = BitConverter.ToUInt32(setting.current, 0);
            return new { profile=baseProfile ? "base" : "current_global", status=status, setting_id="0x10ECECC9",
                setting_name=setting.name, value=current, predefined_value=BitConverter.ToUInt32(setting.predefined, 0),
                location=setting.location, current_is_predefined=setting.isCurrentPredefined,
                interpretation=current == 1 ? "Prefer no sysmem fallback" : current == 2 ? "Prefer sysmem fallback" : "Default" };
        } finally { Function<SessionCall>(0xdad9cff8)(session); }
    }
    public static void PreferNoFallback() {
        int size = Marshal.SizeOf(typeof(Setting));
        if (size != 12320) throw new Exception("Unexpected NVDRS_SETTING_V1 size: " + size);
        Check(Function<Initialize>(0x0150e828)(), "Initialize");
        StringBuilder settingName = new StringBuilder(2048);
        Check(Function<SettingName>(0xd61cbe6e)(0x10ececc9, settingName), "Resolve fallback setting ID");
        IntPtr session;
        Check(Function<Create>(0x0694d52e)(out session), "CreateSession");
        try {
            Check(Function<SessionCall>(0x375dbd6b)(session), "LoadSettings");
            foreach (uint profileFunction in new uint[] { 0xda8466a0u, 0x617bff9fu }) {
                IntPtr profile;
                Check(Function<ProfileCall>(profileFunction)(session, out profile), "GetProfile");
                Setting setting = new Setting { version=(uint)(size | (1 << 16)), name="",
                    predefined=new byte[4100], current=new byte[4100] };
                int status = Function<GetSetting>(0x73bf8338)(session, profile, 0x10ececc9, ref setting);
                if (status != 0 && status != -160) Check(status, "GetSetting before update");
                setting.id=0x10ececc9;
                setting.type=0;
                setting.isCurrentPredefined=0;
                setting.current=BitConverter.GetBytes((uint)1);
                Array.Resize(ref setting.current, 4100);
                Check(Function<SetSetting>(0x577dd202)(session, profile, ref setting), "SetSetting");
            }
            Check(Function<SessionCall>(0xfcbc7e14)(session), "SaveSettings");
        } finally { Function<SessionCall>(0xdad9cff8)(session); }
    }
    public static object StagePrivateWithoutSave() {
        return StageSetterWithoutSave(false, 0x10ececc9);
    }
    public static object StageSetterWithoutSave(bool baseProfile, uint settingId) {
        int size = Marshal.SizeOf(typeof(Setting));
        if (size != 12320) throw new Exception("Unexpected NVDRS_SETTING_V1 size: " + size);
        Check(Function<Initialize>(0x0150e828)(), "Initialize");
        IntPtr session;
        Check(Function<Create>(0x0694d52e)(out session), "CreateSession");
        try {
            Check(Function<SessionCall>(0x375dbd6b)(session), "LoadSettings");
            IntPtr profile;
            Check(Function<ProfileCall>(baseProfile ? 0xda8466a0u : 0x617bff9fu)(session, out profile), "GetProfile");
            Setting before = new Setting { version=(uint)(size | (1 << 16)), name="",
                predefined=new byte[4100], current=new byte[4100] };
            int beforeStatus = Function<GetSetting>(0x73bf8338)(session, profile, settingId, ref before);
            bool publicControl = settingId == 0x1057eb71;
            if (publicControl && (beforeStatus != 0 || before.type != 0))
                return new { profile=baseProfile ? "base" : "current_global", setting_id="0x" + settingId.ToString("X8"),
                    before_read_status=beforeStatus, skipped="Public control must have a readable DWORD value", save_settings_called=false };
            uint value = publicControl ? BitConverter.ToUInt32(before.current, 0) : 1u;
            Setting setting = new Setting { version=(uint)(size | (1 << 16)), name="",
                id=settingId, type=0, location=0, predefined=new byte[4100], current=new byte[4100] };
            Array.Copy(BitConverter.GetBytes(value), setting.current, 4);
            int setStatus = Function<SetSetting>(0x577dd202)(session, profile, ref setting);
            Setting readback = new Setting { version=(uint)(size | (1 << 16)), name="",
                predefined=new byte[4100], current=new byte[4100] };
            int readStatus = Function<GetSetting>(0x73bf8338)(session, profile, settingId, ref readback);
            return new { profile=baseProfile ? "base" : "current_global", setting_id="0x" + settingId.ToString("X8"),
                before_read_status=beforeStatus, attempted_value=value, same_value_public_control=publicControl,
                set_status=setStatus, session_read_status=readStatus,
                session_value=readStatus == 0 ? (uint?)BitConverter.ToUInt32(readback.current, 0) : null,
                save_settings_called=false };
        } finally { Function<SessionCall>(0xdad9cff8)(session); }
    }
}
'@
$result = [ordered]@{ recorded_at=(Get-Date).ToUniversalTime().ToString("o"); operation="read_only"; registered_names=[CudaFallbackPolicyReader]::Names(); profiles=@([CudaFallbackPolicyReader]::Read($false), [CudaFallbackPolicyReader]::Read($true)) }
if ($ValidateSetterControls) {
    if ($PreferNoFallback -or $ValidatePrivateSetting) { throw "Setter-control validation must run without other update/validation modes" }
    $result.operation = "validate_setter_controls_without_save"
    $result.unsaved_sessions = @(
        [CudaFallbackPolicyReader]::StageSetterWithoutSave($false, 0x10ececc9),
        [CudaFallbackPolicyReader]::StageSetterWithoutSave($true, 0x10ececc9),
        [CudaFallbackPolicyReader]::StageSetterWithoutSave($false, 0x1057eb71),
        [CudaFallbackPolicyReader]::StageSetterWithoutSave($true, 0x1057eb71)
    )
    $result.profiles_after_sessions_destroyed = @([CudaFallbackPolicyReader]::Read($false), [CudaFallbackPolicyReader]::Read($true))
}
if ($EnumeratePolicyNames) { $result.available_policy_names = [CudaFallbackPolicyReader]::PolicyNames() }
if ($ValidatePrivateSetting) {
    if ($PreferNoFallback) { throw "Validation-only and persisted update modes must be separate" }
    $result.operation = "stage_private_setting_without_save"
    $result.unsaved_session = [CudaFallbackPolicyReader]::StagePrivateWithoutSave()
    $result.profiles_after_session_destroyed = @([CudaFallbackPolicyReader]::Read($false), [CudaFallbackPolicyReader]::Read($true))
}
if ($PreferNoFallback) {
    $result.operation = "set_prefer_no_sysmem_fallback"
    $result.before = $result.profiles
    [CudaFallbackPolicyReader]::PreferNoFallback()
    $result.profiles = @([CudaFallbackPolicyReader]::Read($false), [CudaFallbackPolicyReader]::Read($true))
    foreach ($profile in $result.profiles) {
        if ($profile.status -ne 0 -or $profile.value -ne 1) { throw "Fallback policy write did not verify" }
    }
}
$result | ConvertTo-Json -Depth 6 -Compress
