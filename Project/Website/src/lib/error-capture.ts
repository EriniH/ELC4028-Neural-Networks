let lastCapturedError: unknown;

function capture(error: unknown) {
  lastCapturedError = error;
}

const globalScope = globalThis as typeof globalThis & {
  addEventListener?: (type: string, listener: (event: unknown) => void) => void;
  process?: { on?: (event: string, listener: (error: unknown) => void) => void };
};

globalScope.addEventListener?.("error", (event) => {
  const errorEvent = event as { error?: unknown; message?: unknown };
  capture(errorEvent.error ?? errorEvent.message ?? event);
});

globalScope.addEventListener?.("unhandledrejection", (event) => {
  const rejectionEvent = event as { reason?: unknown };
  capture(rejectionEvent.reason ?? event);
});

globalScope.process?.on?.("uncaughtException", capture);
globalScope.process?.on?.("unhandledRejection", capture);

export function consumeLastCapturedError() {
  const captured = lastCapturedError;
  lastCapturedError = undefined;
  return captured;
}
