// next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    serverActions: {
      bodySizeLimit: '5000mb',
    },
  },
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.externals = config.externals || [];
      config.externals.push('ffmpeg-static', 'fluent-ffmpeg', 'ffprobe-static');
    }
    return config;
  },
};

export default nextConfig;
