"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs");
var fs = require("fs"); //sirve para leer archivos
var returned = fs.readFileSync("ex1data1.txt", "utf8"); //le digo el archivo que tiene q leer y el formato q tiene
var filas = returned.split('\n'); //separo el texto por filas
var array = [[], []];
for (var i = 0; i < filas.length - 1; i++) {
    var columnas = filas[i].split(','); //separo los datos en columnas segun las ","
    array[i] = columnas;
    //console.log(columnas[0]);
}
var x = [];
var y = [];
array = array;
//console.log(typeof(array[0][0]));
for (var i = 0; i < array.length; i++) {
    x[i] = array[i][0];
    y[i] = array[i][1];
}
//console.log(x);
var tensorX = tf.tensor(x, [x.length, 1], "float32");
var tensorY = tf.tensor(y, [y.length, 1], "float32");
var model = tf.sequential();
var layer = tf.layers.dense({ units: 1, inputShape: [1] });
model.add(layer);
model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
model.fit(tensorX, tensorY);
console.log(model.predict(tf.tensor2d([[5], [1]], [2, 1])).toString());
