/// Types for the broadcast service handler, which `broadcastContract.test.ts`
/// imports across the package boundary to test the real service against the
/// real client (ADR-182). The service is plain ESM with no build step, so it
/// ships no declarations; this describes only the surface the contract test
/// touches rather than pulling `@types/node` into the app.
declare module "*/services/musica-broadcast/handler.mjs" {
  export interface ServiceRequest {
    store: unknown;
    subject: string;
    method: string;
    path: string;
    query?: Record<string, unknown>;
    body?: unknown;
    nowSeconds: number;
  }

  export interface ServiceResponse {
    status: number;
    // Intentionally loose: the point of the contract test is to feed whatever
    // the service really returns through the client's own normalizers, so
    // typing this precisely here would assume the answer.
    body: any;
  }

  export function handle(request: ServiceRequest): Promise<ServiceResponse>;
  export function broadcastIdFor(subject: string, secret?: string): string;
}
