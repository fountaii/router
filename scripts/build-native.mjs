import { spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readdirSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const root = fileURLToPath(new URL('../', import.meta.url));
const native = path.join(root, 'src/native');
const target = `${process.platform}/${process.arch}`;
const bin = path.join(root, 'src/onnx/bin', target);
const build = path.join(root, '.cache/native', target);
const output = path.join(build, 'onnxruntime_binding.node');
const includes = [path.join(native, 'include'), path.join(native, 'include/node'),
  path.dirname(require.resolve('node-addon-api/package.json'))];
const sources = readdirSync(native).filter(file => file.endsWith('.cc')).map(file => path.join(native, file));
const definitions = ['NAPI_VERSION=8', 'NAPI_CPP_EXCEPTIONS', 'ORT_API_MANUAL_INIT'];
const runtimeName = process.platform === 'win32' ? 'onnxruntime.dll'
  : process.platform === 'linux' ? 'libonnxruntime.so.1' : 'libonnxruntime.1.dylib';
const localRuntime = path.join(native, 'lib', target, runtimeName);
const bundledRuntime = path.join(bin, runtimeName);
const runtime = existsSync(localRuntime) ? localRuntime : bundledRuntime;

function run(command, args, options = {}) {
  const result = spawnSync(command, args, { cwd: build, stdio: 'inherit', ...options });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} failed (${result.status ?? result.signal}).`);
}

mkdirSync(build, { recursive: true });
if (!existsSync(runtime)) {
  throw new Error(`Missing local ONNX Runtime library. Place ${runtimeName} in ${path.dirname(localRuntime)}. Headers alone cannot build the runtime library.`);
}
mkdirSync(bin, { recursive: true });

if (process.platform === 'win32') {
  if (!['x64', 'arm64'].includes(process.arch)) throw new Error(`Unsupported architecture: ${process.arch}`);
  const vswhere = path.join(process.env['ProgramFiles(x86)'] ?? 'C:/Program Files (x86)',
    'Microsoft Visual Studio/Installer/vswhere.exe');
  const result = spawnSync(vswhere, ['-latest', '-prerelease', '-products', '*',
    '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'],
    { encoding: 'utf8' });
  const installation = result.stdout?.trim();
  if (!installation) throw new Error('Install Visual Studio Build Tools with Desktop development with C++ and Windows SDK.');
  // Only OrtGetApiBase is imported; all other ORT calls go through its API table.
  writeFileSync(path.join(build, 'onnxruntime.def'), 'LIBRARY onnxruntime.dll\nEXPORTS\n  OrtGetApiBase\n');
  const quote = value => `"${value}"`;
  const args = ['/nologo', '/LD', '/O2', '/EHsc', '/std:c++17', '/MD', '/utf-8', '/DHOST_BINARY=\\"node.exe\\"',
    ...definitions.map(value => `/D${value}`), ...includes.map(value => `/I${quote(value)}`),
    ...sources.map(quote), quote(path.join(native, 'lib/win32', process.arch, 'node.lib')),
    quote(path.join(build, 'onnxruntime.lib')), '/link', 'delayimp.lib', '/DELAYLOAD:node.exe', `/OUT:${quote(output)}`];
  writeFileSync(path.join(build, 'compile.rsp'), args.join(' '));
  const batch = path.join(build, 'build.cmd');
  writeFileSync(batch, [
    '@echo off',
    `call ${quote(path.join(installation, 'VC/Auxiliary/Build/vcvarsall.bat'))} ${process.arch === 'arm64' ? 'x64_arm64' : 'x64'}`,
    'if errorlevel 1 exit /b %errorlevel%',
    `lib /nologo /def:onnxruntime.def /machine:${process.arch} /out:onnxruntime.lib`,
    'if errorlevel 1 exit /b %errorlevel%',
    'cl @compile.rsp',
    'exit /b %errorlevel%', '',
  ].join('\r\n'));
  run(process.env.ComSpec ?? 'cmd.exe', ['/d', '/c', batch]);
} else if (['linux', 'darwin'].includes(process.platform)) {
  run(process.env.CXX ?? 'c++', ['-std=c++17', '-O2', '-fPIC', '-shared', '-pthread',
    ...definitions.map(value => `-D${value}`), ...includes.flatMap(value => ['-I', value]),
    ...sources, runtime,
    ...(process.platform === 'linux' ? ['-Wl,-rpath,$ORIGIN'] : ['-undefined', 'dynamic_lookup', '-Wl,-rpath,@loader_path']),
    '-o', output]);
} else {
  throw new Error(`Unsupported platform: ${process.platform}`);
}

// Check loading in a separate process before replacing the working addon.
copyFileSync(runtime, path.join(build, runtimeName));
run(process.execPath, ['-e', `const b = require(${JSON.stringify(output)}); if (b.listSupportedBackends()[0]?.name !== 'cpu') throw Error('Invalid binding');`]);
if (runtime !== bundledRuntime) copyFileSync(runtime, bundledRuntime);
copyFileSync(output, path.join(bin, 'onnxruntime_binding.node'));
console.log(`Built src/onnx/bin/${target}/onnxruntime_binding.node`);
