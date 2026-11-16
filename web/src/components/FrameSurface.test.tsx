import { render, screen, waitFor } from "@testing-library/react";

import { FrameSurface, parseFrameHeaders } from "./FrameSurface";

function frameResponse(index = 1, publishedAt = Date.now() / 1000): Response {
  return new Response(new Blob([`frame-${index}`], { type: "image/jpeg" }), {
    status: 200,
    headers: {
      "Content-Type": "image/jpeg",
      "X-Frame-Index": String(index),
      "X-Frame-Published-At": String(publishedAt),
    },
  });
}

function installCanvas(drawImage = vi.fn()) {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
    drawImage,
  } as unknown as CanvasRenderingContext2D);
  return drawImage;
}

test("treats missing or legacy frame timestamps as unavailable", () => {
  expect(parseFrameHeaders(new Headers())).toEqual({
    frameIndex: null,
    publishedAt: null,
  });
  expect(
    parseFrameHeaders(
      new Headers({
        "X-Frame-Index": "7",
        "X-Frame-Published-At": "0",
      }),
    ),
  ).toEqual({ frameIndex: 7, publishedAt: null });
});

test("keeps one canvas mounted and one frame request in flight", async () => {
  installCanvas();
  let resolveRequest: ((response: Response) => void) | undefined;
  const pending = new Promise<Response>((resolve) => {
    resolveRequest = resolve;
  });
  vi.stubGlobal(
    "fetch",
    vi.fn(() => pending),
  );
  vi.stubGlobal(
    "createImageBitmap",
    vi.fn(async () => ({ width: 640, height: 480, close: vi.fn() })),
  );

  const view = render(
    <FrameSurface endpoint="/api/frame" streamKey="camera" label="Camera" />,
  );
  const canvas = await screen.findByRole("img", { name: "Camera" });
  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));

  view.rerender(
    <FrameSurface
      endpoint="/api/frame"
      streamKey="camera"
      label="Camera live"
    />,
  );

  expect(screen.getByRole("img", { name: "Camera live" })).toBe(canvas);
  expect(fetch).toHaveBeenCalledTimes(1);
  resolveRequest?.(frameResponse());
});

test("recovers after an initial missing frame", async () => {
  const drawImage = installCanvas();
  vi.stubGlobal(
    "fetch",
    vi
      .fn()
      .mockResolvedValueOnce(new Response(null, { status: 404 }))
      .mockResolvedValue(frameResponse()),
  );
  vi.stubGlobal(
    "createImageBitmap",
    vi.fn(async () => ({ width: 640, height: 480, close: vi.fn() })),
  );

  render(
    <FrameSurface
      endpoint="/api/frame"
      streamKey="camera"
      label="Camera"
      targetFps={120}
    />,
  );

  await waitFor(() => expect(drawImage).toHaveBeenCalled());
  expect(fetch).toHaveBeenCalledTimes(2);
});

test("cancels a stale decoded frame when the stream changes", async () => {
  const drawImage = installCanvas();
  let resolveOld: ((bitmap: ImageBitmap) => void) | undefined;
  const oldBitmap = { width: 640, height: 480, close: vi.fn(), id: "old" };
  const newBitmap = { width: 640, height: 480, close: vi.fn(), id: "new" };
  const oldPending = new Promise<ImageBitmap>((resolve) => {
    resolveOld = resolve;
  });
  vi.stubGlobal(
    "fetch",
    vi.fn(() => Promise.resolve(frameResponse())),
  );
  vi.stubGlobal(
    "createImageBitmap",
    vi
      .fn()
      .mockReturnValueOnce(oldPending)
      .mockResolvedValue(newBitmap as unknown as ImageBitmap),
  );

  const view = render(
    <FrameSurface endpoint="/api/old" streamKey="old" label="Preview" />,
  );
  await waitFor(() => expect(createImageBitmap).toHaveBeenCalledTimes(1));
  view.rerender(
    <FrameSurface endpoint="/api/new" streamKey="new" label="Preview" />,
  );
  await waitFor(() => expect(drawImage).toHaveBeenCalled());
  resolveOld?.(oldBitmap as unknown as ImageBitmap);
  await Promise.resolve();

  expect(drawImage).toHaveBeenCalledTimes(1);
  expect(drawImage.mock.calls[0][0]).toBe(newBitmap);
  expect(oldBitmap.close).toHaveBeenCalled();
});
