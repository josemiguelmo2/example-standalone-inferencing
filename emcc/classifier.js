const Module = require('../build/emcc/edge-impulse-standalone');

let initialized = false;
Module.onRuntimeInitialized = function() {
    initialized = true;
};

class EdgeImpulseClassifier {
    _initialized = false;

    constructor() {

    }

    init() {
        if (initialized === true) return Promise.resolve();

        return new Promise((resolve) => {
            Module.onRuntimeInitialized = () => {
                resolve();
                initialized = true;
            };
        });
    }

    classify(rawData, debug = false) {
        if (!initialized) throw new Error('Module is not initialized');

        const obj = this._arrayToHeap(rawData);

        let ret = Module.run_classifier(obj.byteOffset, rawData.length, debug);

        Module._free(obj);

        if (ret.result !== 0) {
            throw new Error('Classification failed (err code: ' + ret.result + ')');
        }

        let jsResult = {
            anomaly: ret.anomaly,
            results: []
        };

        for (let cx = 0; cx < ret.classification.size(); cx++) {
            let c = ret.classification.get(cx);
            jsResult.results.push({ label: c.label, value: c.value });
        }

        return jsResult;
    }

    _arrayToHeap(data) {
        let typedArray = new Float32Array(data);
        let numBytes = typedArray.length * typedArray.BYTES_PER_ELEMENT;
        let ptr = Module._malloc(numBytes);
        let heapBytes = new Uint8Array(Module.HEAPU8.buffer, ptr, numBytes);
        heapBytes.set(new Uint8Array(typedArray.buffer));
        return heapBytes;
    }
}

module.exports = EdgeImpulseClassifier;
