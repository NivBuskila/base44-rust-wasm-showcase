import type { Plugin } from 'vite';

/**
 * Dev-server drop box for QA session records.
 *
 * `web/src/qa-recorder.ts` keeps its summary in `localStorage`, which was meant
 * to let a session driven in a standalone tab be read back from any context on
 * the same origin. That does not work: Chrome partitions storage by top-level
 * site, so the embedded preview iframe — same origin, different top-level site —
 * gets its own bucket and can never see the standalone tab's record. Storage is
 * therefore useless as a bridge out of a human QA pass.
 *
 * An HTTP round trip through the dev server has no such partition. The tab
 * `POST`s its summary here, the record is held in memory, and anything with
 * shell access reads it back with a plain `GET` — no browser, no storage, no
 * iframe.
 *
 * In memory on purpose: this is diagnostics about the process currently running,
 * so it should die with the dev server rather than leave a stale file behind
 * that looks like a fresh measurement. Only the newest record is kept, because
 * the interesting session is the one just driven.
 *
 * Dev-only. A production build has no server here, and the recorder keeps
 * working against `localStorage` regardless.
 */

/** Where the tab posts and the shell reads. */
export const QA_SESSION_ENDPOINT = '/__qa/session';

/** Refuses to buffer more than this, so a bad client cannot exhaust memory. */
const MAX_BODY_BYTES = 256 * 1024;

/** The Node request/response surface this needs, typed structurally because
 * the project deliberately carries no Node typings. */
interface Req {
  url?: string;
  method?: string;
}
/** The stream half of the request. Kept off {@link Req} because Vite types the
 * middleware against Node's `IncomingMessage`, which without Node typings
 * resolves to a shape that has no `on` — so requiring it there makes the
 * handler unassignable. */
type ReqStream = { on(event: string, listener: (chunk?: unknown) => void): void };
interface Res {
  statusCode: number;
  setHeader(name: string, value: string): void;
  end(body?: string): void;
}

export function qaSessionEndpoint(): Plugin {
  let record: { receivedAt: string; session: unknown } | null = null;

  return {
    name: 'aether:qa-session-endpoint',
    configureServer(server) {
      // `unknown` parameters, narrowed below: Vite types this against Node's
      // `IncomingMessage`/`ServerResponse`, and with no Node typings in the
      // project those resolve to shapes that share no properties with any
      // structural type written here, which makes an annotated handler
      // unassignable.
      server.middlewares.use((rawReq: unknown, rawRes: unknown, next: () => void) => {
        const req = rawReq as Req;
        const res = rawRes as Res;
        if (req.url?.split('?')[0] !== QA_SESSION_ENDPOINT) return next();

        if (req.method === 'GET') {
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify(record ?? { receivedAt: null, session: null }));
          return;
        }

        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.end();
          return;
        }

        const stream = req as unknown as ReqStream;
        let body = '';
        let tooLarge = false;
        stream.on('data', (chunk?: unknown) => {
          if (tooLarge) return;
          body += String(chunk);
          if (body.length > MAX_BODY_BYTES) {
            tooLarge = true;
            res.statusCode = 413;
            res.end();
          }
        });
        stream.on('end', () => {
          if (tooLarge) return;
          try {
            record = { receivedAt: new Date().toISOString(), session: JSON.parse(body) };
            res.statusCode = 204;
            res.end();
          } catch {
            res.statusCode = 400;
            res.end();
          }
        });
      });
    },
  };
}
