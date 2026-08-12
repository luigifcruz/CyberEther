addToLibrary({
  _emval_await: () => {
    throw new Error('_emval_await requires an async Emscripten build');
  },

  _wasmfs_opfs_read_access__i53abi: true,
  _wasmfs_opfs_read_access__deps: ['$wasmfsOPFSAccessHandles'],
  _wasmfs_opfs_read_access: (accessID, bufPtr, len, pos) => {
    const offset = Number(pos);
    if (!Number.isSafeInteger(offset)) {
      return -28;
    }

    const accessHandle = wasmfsOPFSAccessHandles.get(accessID);
    const data = new Uint8Array(len);
    try {
      const bytesRead = accessHandle.read(data, { at: offset });
      HEAPU8.set(data.subarray(0, bytesRead), bufPtr);
      return bytesRead;
    } catch (error) {
      if (error.name === 'TypeError') {
        return -28;
      }
      console.error('OPFS read failed:', error);
      return -29;
    }
  },

  _wasmfs_opfs_write_access__i53abi: true,
  _wasmfs_opfs_write_access__deps: ['$wasmfsOPFSAccessHandles'],
  _wasmfs_opfs_write_access: (accessID, bufPtr, len, pos) => {
    const offset = Number(pos);
    if (!Number.isSafeInteger(offset)) {
      return -28;
    }

    const accessHandle = wasmfsOPFSAccessHandles.get(accessID);
    // OPFS rejects views backed by pthreads' SharedArrayBuffer memory.
    const data = HEAPU8.slice(bufPtr, bufPtr + len);
    try {
      return accessHandle.write(data, { at: offset });
    } catch (error) {
      if (error.name === 'TypeError') {
        return -28;
      }
      console.error('OPFS write failed:', error);
      return -29;
    }
  },
});
