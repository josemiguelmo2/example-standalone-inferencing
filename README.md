# Edge Impulse Example: stand-alone inferencing

This runs an exported impulse locally on your machine. See the documentation at [Running your impulse locally](https://docs.edgeimpulse.com/docs/running-your-impulse-locally).

## WebAssembly (Node.js)

This is an experimental version of the standalone inferencer that compiles into WebAssembly, so you can run the application from Node.js or a web browser. To compile the application install Emscripten and then run:

```
$ make -f Makefile.emcc
```

Then create a file `test.js` with the following content:

```js
const Classifier = require('./emcc/classifier');

(async () => {
    try {
        let classifier = new Classifier();
        await classifier.init();

        let result = classifier.classify([
            /* paste your raw data here */
        ]);

        console.log(result);
    }
    catch (ex) {
        console.error('Failed to classify', ex);
    }
})();
```

And run the application with Node.js:

```
$ node test.js
```

## WebAssembly (browser)

You can also run the application from the browser:

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Emscripten example</title>
</head>

<body>
    <script src="../build/emcc/edge-impulse-standalone.js"></script>
    <script>
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

        (async () => {
            try {
                let classifier = new EdgeImpulseClassifier();
                await classifier.init();

                console.time('classify');
                let result = classifier.classify([
                    /* your raw data */
                ]);
                console.timeEnd('classify');

                console.log(result);
            }
            catch (ex) {
                console.error('Failed to classify', ex);
            }
        })();
    </script>
</body>
</html>
```
