import * as tf from "@tensorflow/tfjs";
import * as fs from "fs";//sirve para leer archivos
let returned = fs.readFileSync("ex1data1.txt", "utf8");//le digo el archivo que tiene q leer y el formato q tiene


let filas = returned.split('\n');//separo el texto por filas
let array: any[][] = [[], []];
for (let i = 0; i < filas.length - 1; i++) {
    let columnas: any[] = filas[i].split(',');//separo los datos en columnas segun las ","

    array[i] = columnas;
    //console.log(columnas[0]);

}

let x: number[]=[];
let y: number[]=[];
array=array as number[][];
//console.log(typeof(array[0][0]));
for (let i = 0; i < array.length; i++) {
        x[i]=array[i][0];
        y[i]=array[i][1];
}
//console.log(x);

let tensorX=tf.tensor(x,[x.length,1],"float32");
let tensorY=tf.tensor(y,[y.length,1],"float32");
let model=tf.sequential();

let layer=tf.layers.dense({units:1,inputShape:[1]});

model.add(layer);
model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });


model.fit(tensorX,tensorY);


console.log(model.predict(tf.tensor2d([[5],[1]], [2, 1])).toString());
