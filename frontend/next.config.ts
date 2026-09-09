import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /**
   * anime.js's text splitter rewrites an element's children into measured line
   * boxes and does not survive StrictMode's deliberate mount/unmount/remount:
   * the second pass measures what the first already rewrote, so headlines come
   * back duplicated or collapsed. Development-only — the built output is
   * correct — and turned off so development shows what actually ships.
   */
  reactStrictMode: false,
};

export default nextConfig;
