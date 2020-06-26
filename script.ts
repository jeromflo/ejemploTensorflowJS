import * as tf from "@tensorflow/tfjs";
import * as fs from "fs";//sirve para leer archivos
let returned = fs.readFileSync("ex1data1.txt", "utf8");//le digo el archivo que tiene q leer y el formato q tiene


let filas = returned.split('\n');//separo el texto por filas
let array: any[][] = [[], []];
for (let i = 0; i < filas.length - 1; i++) {
    let columnas: any[] = filas[i].split(',');//separo los datos en columnas segun las ","

    array[i] = columnas;

}

let x: number[]=[];
let y: number[]=[];
for (let i = 0; i < array.length; i++) {
        x[i]=array[i][0];
        y[i]=array[i][1];
}

let tensorX=tf.tensor(x,[x.length,1],"float32");//creo el tensor para la variable x
let tensorY=tf.tensor(y,[y.length,1],"float32");//creo un tensor para la variable y
//en ambos casos le digo el shape, es decir el tamanyo de la matriz y el tipo que queremos
let model=tf.sequential();
//creamos nuestro modelo

let layer=tf.layers.dense({units:1,inputShape:[1]});
//creamos nueestra capa de neuronas, en este caso, le decimos que queremos una neurona con una de entrada
model.add(layer);
//anyadimos la capa al modelo
model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
//compilamos con la funcion de error de error medio y el optimizador sgd

model.fit(tensorX,tensorY);
//entrenamos el modelo


console.log(model.predict(tf.tensor2d([[5],[1]], [2, 1])).toString());
//imprimimos la prediccion segun los valores pasados
