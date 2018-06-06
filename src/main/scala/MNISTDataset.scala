package main.scala

import java.nio.file.{Files, Paths}

import breeze.linalg.{DenseMatrix, rot90}

object MNISTDataset {

  lazy val trainingImages: Iterator[DenseMatrix[Byte]] =
    Files.readAllBytes(Paths.get("/Users/benikm91/Documents/Programming/Private/Scala/RestrictedBoltzmannMaschine/src/main/resources/train-images-idx3-ubyte"))
      .drop(16)
      .grouped(28*28) //.map(_.grouped(28).toArray.reverse).map(_.flatten)
      .map(data => rot90(new DenseMatrix[Byte](28, 28, data)))

  lazy val trainingLabels: Iterator[Byte] =
    Files.readAllBytes(Paths.get("/Users/benikm91/Documents/Programming/Private/Scala/RestrictedBoltzmannMaschine/src/main/resources/train-labels-idx1-ubyte"))
      .drop(8).toIterator

  lazy val trainingData: Iterator[(Byte, DenseMatrix[Double])] = trainingLabels.zip(trainingImages.map(_.map(x => (math.abs(x.toDouble)) / 128)))

}
