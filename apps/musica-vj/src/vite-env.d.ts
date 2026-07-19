/// <reference types="vite/client" />

interface Navigator {
  requestMIDIAccess?: (options?: { sysex?: boolean; software?: boolean }) => Promise<MIDIAccess>;
}
