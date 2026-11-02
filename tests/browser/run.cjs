const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const {parseArgs} = require('node:util');

async function enumerate(context) {
    const page = await context.newPage();
    try {
        await page.goto('chrome://usb-internals');
        return await page.evaluate(async () => {
            const {UsbInternalsPageHandler} = await import('./usb_internals.mojom-webui.js');
            const {UsbDeviceManagerRemote} = await import('./usb_manager.mojom-webui.js');
            const manager = new UsbDeviceManagerRemote();
            await UsbInternalsPageHandler.getRemote().bindUsbDeviceManagerInterface(
                manager.$.bindNewPipeAndPassReceiver());
            return (await manager.getDevices(null)).results.map(device => ({
                vendor_id: device.vendorId, product_id: device.productId,
                name: device.productName, serial: device.serialNumber,
            }));
        });
    } finally {
        await page.close();
    }
}

async function main() {
    const {values, positionals} = parseArgs({
        allowPositionals: true,
        options: {usb: {type: 'boolean'}, timeout: {type: 'string', default: '150000'}},
    });
    const [binary, ...testArgs] = positionals;
    const timeout = Number(values.timeout);
    if (!binary) {
        throw new Error('Usage: node run.cjs <test.js> [--usb] [--timeout ms] [-- test arguments]');
    }
    if (!Number.isFinite(timeout) || timeout <= 0) throw new Error('Invalid test timeout');
    const modulePath = path.resolve(binary);
    const directory = path.dirname(modulePath);
    for (const file of [modulePath, path.join(directory, `${path.parse(modulePath).name}.wasm`)]) {
        if (!fs.existsSync(file)) {
            throw new Error(`WASM test build is incomplete: ${file}. Compile the test target before using --no-rebuild.`);
        }
    }
    let chromium;
    try {
        ({chromium} = require('playwright'));
    } catch {
        throw new Error('Install the runner: npm ci --prefix tests/browser');
    }
    const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'cyberether-wasm-'));
    const launch = {
        headless: true,
        channel: 'chromium',
        args: ['--enable-unsafe-webgpu'],
        ...(process.env.SDR_WASM_CHROMIUM ? {executablePath: process.env.SDR_WASM_CHROMIUM} : {}),
    };
    let context;
    let timer;
    const server = http.createServer((request, response) => {
        response.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
        response.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
        const name = new URL(request.url, 'http://localhost').pathname;
        if (name === '/favicon.ico') {
            response.writeHead(204);
            response.end();
            return;
        }
        if (name === '/') {
            response.setHeader('Content-Type', 'text/html');
            response.end('<!doctype html><meta charset="utf-8"><title>CyberEther WASM tests</title><canvas id="canvas" width="1280" height="720"></canvas>');
            return;
        }
        const file = name === '/module.js' ? modulePath : path.join(directory, path.basename(name));
        if (!fs.existsSync(file) || !fs.statSync(file).isFile()) {
            response.writeHead(404);
            response.end();
            return;
        }
        response.setHeader('Content-Type', file.endsWith('.wasm') ? 'application/wasm' : 'text/javascript');
        fs.createReadStream(file).pipe(response);
    });
    try {
        await new Promise((resolve, reject) => {
            server.once('error', reject);
            server.listen(0, '127.0.0.1', resolve);
        });
        const origin = `http://127.0.0.1:${server.address().port}`;
        if (values.usb) {
            context = await chromium.launchPersistentContext(profile, launch);
            const devices = await enumerate(context);
            await context.close();
            context = null;
            const preferencesPath = path.join(profile, 'Default', 'Preferences');
            const preferences = JSON.parse(fs.readFileSync(preferencesPath, 'utf8'));
            preferences.profile ||= {};
            preferences.profile.managed_web_usb_allow_devices_for_urls = [{
                devices: devices.map(({vendor_id, product_id}) => ({vendor_id, product_id})),
                urls: [origin],
            }];
            fs.writeFileSync(preferencesPath, JSON.stringify(preferences));
            console.log(`[WASM] ${devices.length} USB devices discovered headlessly`);
        }
        context = await chromium.launchPersistentContext(profile, launch);
        if (values.usb) await enumerate(context);
        const page = await context.newPage();
        let rejectFailure;
        const failure = new Promise((_, reject) => { rejectFailure = reject; });
        failure.catch(() => {});
        timer = setTimeout(() => rejectFailure(new Error('WASM test timed out')), timeout);
        page.on('console', message => console.log(message.text()));
        page.on('pageerror', rejectFailure);
        page.on('crash', () => rejectFailure(new Error('Chromium test page crashed')));
        await page.goto(origin);
        await Promise.race([page.evaluate(async testArgs => {
            window.testExit = null;
            window.testFailure = null;
            const fail = error => {
                window.testFailure = String(error?.stack || error);
                window.testExit = 1;
            };
            window.addEventListener('unhandledrejection', event => fail(event.reason));
            try {
                const {default: create} = await import('/module.js');
                const runtime = await create({
                    noInitialRun: true,
                    canvas: document.getElementById('canvas'),
                    print: text => console.log(text),
                    printErr: text => console.error(text),
                    onExit: status => { window.testExit = status; },
                    onAbort: fail,
                });
                const status = runtime.callMain(testArgs);
                if (status) fail(new Error(`WASM main thread failed to start (${status})`));
            } catch (error) { fail(error); }
        }, testArgs), failure]);
        await Promise.race([page.waitForFunction(() => window.testExit !== null, null, {timeout}), failure]);
        const result = await page.evaluate(() => ({status: window.testExit, error: window.testFailure}));
        if (result.error) throw new Error(result.error);
        return result.status;
    } finally {
        clearTimeout(timer);
        if (context) await context.close();
        await new Promise(resolve => server.close(resolve));
        fs.rmSync(profile, {recursive: true, force: true});
    }
}

main().then(status => { process.exitCode = status; }).catch(error => {
    console.error(error.stack || error);
    process.exitCode = 1;
});
