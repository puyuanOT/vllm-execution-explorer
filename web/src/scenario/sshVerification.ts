import type { EngineSettings } from "./types.ts"

export interface SshVerification {
  user: string
  host: string
  port: string
  automaticSetup: boolean
  identity: File
  runnerEndpoint: string
}

export function createSshVerification(
  settings: EngineSettings,
  identity: File,
  runnerEndpoint: string,
): SshVerification {
  return {
    user: settings.sshUser.trim(),
    host: settings.sshHost.trim(),
    port: settings.sshPort.trim(),
    automaticSetup: settings.automaticSetup,
    identity,
    runnerEndpoint,
  }
}

export function sshVerificationError(
  settings: EngineSettings,
  identity: File | null,
  verification: SshVerification | null,
): string {
  if (settings.hostKind !== "ssh") return ""
  if (!identity) return "Choose the SSH private key and check this host before capture."
  if (!verification) return "Check this SSH host before capture."
  if (
    verification.user !== settings.sshUser.trim()
    || verification.host !== settings.sshHost.trim()
    || verification.port !== settings.sshPort.trim()
    || verification.automaticSetup !== settings.automaticSetup
    || verification.identity !== identity
    || verification.runnerEndpoint !== settings.runnerUrl
  ) {
    return "The SSH host or private key changed. Check the host again before capture."
  }
  return ""
}

export function remoteRuntimeError(gpu: string | null | undefined, version: string | null | undefined): string {
  if (!gpu?.trim()) return "The execution host did not report an NVIDIA GPU."
  if (!version?.startsWith("0.30.")) {
    return `The execution host needs vLLM 0.30.x; found ${version?.trim() || "no vLLM install"}.`
  }
  return ""
}
