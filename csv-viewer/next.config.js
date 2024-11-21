/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable static exports for GitHub Pages
  output: 'export',
  
  // Required for static exports
  images: {
    unoptimized: true,
  },
  
  // This should match your GitHub repository name
  // Remove this during local development
  basePath: '/bqfgsports',
  
  // React strict mode for better development
  reactStrictMode: true,
  
  // Allow importing SVG files
  webpack(config) {
    config.module.rules.push({
      test: /\.svg$/,
      use: ['@svgr/webpack'],
    })
    return config
  },
}

module.exports = nextConfig