import { render, screen } from "@testing-library/react";

import { StatusBadge } from "./StatusBadge";

test("exposes machine and human readable semantic status", () => {
  render(<StatusBadge status="failed" />);

  const badge = screen.getByRole("status", { name: "failed" });
  expect(badge).toHaveTextContent("Failed");
  expect(badge).toHaveAttribute("data-state", "failed");
});
