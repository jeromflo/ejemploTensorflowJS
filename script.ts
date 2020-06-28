import * as tf from "@tensorflow/tfjs";
//import * as tf from "@tensorflow/tfjs-node";
import * as fs from "fs";//sirve para leer archivos
let returned = fs.readFileSync("ex1data1.txt", "utf8");//le digo el archivo que tiene q leer y el formato q tiene


let filas = returned.split('\n');//separo el texto por filas
let array: any[][] = [[], []];
for (let i = 0; i < filas.length - 1; i++) {
    let columnas: any[] = filas[i].split(',');//separo los datos en columnas segun las ","

    array[i] = columnas;

}

let x: number[] = [];
let y: number[] = [];
for (let i = 0; i < array.length; i++) {
    x[i] = array[i][0];
    y[i] = array[i][1];
}

let tensorX = tf.tensor(x, [x.length, 1], "float32");//creo el tensor para la variable x
let tensorY = tf.tensor(y, [y.length, 1], "float32");//creo un tensor para la variable y
//en ambos casos le digo el shape, es decir el tamanyo de la matriz y el tipo que queremos
let model = tf.sequential();
//creamos nuestro modelo

let layer = tf.layers.dense({ units: 100, inputShape: [1], activation: "relu" });//inputshape, hace referencia, al numero de atributos de los datos
//creamos nueestra capa de neuronas, en este caso, le decimos que queremos 100 neuronas de primera capa oculta 
model.add(layer);
layer = tf.layers.dropout({ rate: 0.2 });//es parecido a una capa densa, pero es una capa intermedia, la cual tiene un 20% de prob de no transmitir la informacion
//haciendo que no se haga mayor overfitting
model.add(layer);
layer = tf.layers.dense({ units: 50, activation: "relu", inputShape: [100] });//siguiente capa oculta con 50 neuronas
model.add(layer);
layer = tf.layers.dense({ units: 100, activation: "relu", inputShape: [50] });//siguiente capa oculta con 100 neuronas
model.add(layer);
layer = tf.layers.dense({ units: 1, activation: "relu" });//capa de salida con una neurona, ya que solo predecimos un valor
model.add(layer);


//anyadimos la capa al modelo
model.compile({ loss: tf.losses.meanSquaredError, optimizer: "adam", metrics: tf.metrics.meanAbsoluteError });
//compilamos con la funcion de error de error medio y el optimizador sgd o adam
model.summary();//metodo que imprime una lista con las capas


 model.fit(tensorX, tensorY, { epochs: 100 }).then((value)=>{//hay que usar promesas, ya que tenemos que esperar a que termine el entranmiento
    //de la red neuronal, permitiendo que el evaluate final, se haga con la ultima iteraccion del fit.
    console.log(value);
    //if(value.epoch.length==14){
    let evaluate=model.evaluate(tensorX, tensorY);
    console.log(evaluate.toString());
    //}
});//epochs es el numero de iteracciones de backpropagetion
 
