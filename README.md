# Edge Impulse Example: stand-alone inferencing

This runs an exported impulse locally on your machine. See the documentation at [Running your impulse locally](https://docs.edgeimpulse.com/docs/running-your-impulse-locally).

## WebAssembly

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
