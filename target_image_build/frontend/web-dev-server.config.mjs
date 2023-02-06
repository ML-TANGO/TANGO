// import { hmrPlugin, presets } from '@open-wc/dev-server-hmr';
import toml from 'markty-toml';
import fs from 'fs';

/** Use Hot Module replacement by adding --hmr to the start command */
const hmr = process.argv.includes('--hmr');

const config = toml(fs.readFileSync('./configs/forklift.toml', 'utf-8'));
const port = config.webui.addr.port;

export default /** @type {import('@web/dev-server').DevServerConfig} */ ({
  open: '/',
  port: port,
  watch: !hmr,
  /** Resolve bare module imports */
  nodeResolve: {
    exportConditions: ['browser', 'development'],
  },
  
  /** Compile JS for older browsers. Requires @web/dev-server-esbuild plugin */
  // esbuildTarget: 'auto'

  /** Set appIndex to enable SPA routing */
  appIndex: 'index.html',

  plugins: [
    /** Use Hot Module Replacement by uncommenting. Requires @open-wc/dev-server-hmr plugin */
    // hmr && hmrPlugin({ exclude: ['**/*/node_modules/**/*'], presets: [presets.litElement] }),
  ],

  // See documentation for all available options
});
