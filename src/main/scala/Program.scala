package main.scala

import java.io._
import java.text.SimpleDateFormat

import scala.util.control.Exception._
import breeze.plot.Figure
import breeze.plot.image
import breeze.linalg._

import scala.util.Random

object Program {

  def debugTrainingData(trainingData: Iterator[(Byte, DenseMatrix[Double])]): Unit = {
    val f2 = Figure()
    for {
      ((_label, _image), index) <- trainingData.drop(new Random().nextInt(100)).take(9).zipWithIndex
    } {
      val plot = f2.subplot(3, 3, index)
      plot.title = _label.toString
      plot += image(_image)
    }
    f2.saveas("image.png")
  }

  private def lala(a: Int): Int = {
    var i: Int = math.sqrt(a).toInt
    while (i > 1) {
      if (a % i == 0) return i
      i -= 1
    }
    1
  }

  def debugTrainedWeights(weights: DenseMatrix[Double], title: String): Unit = {
    val f2 = Figure(title)
    var index = 0
    val extreme = max(weights).toInt + 1
    for (column <- weights(*, ::)) {
      val plot = f2.subplot(math.sqrt(weights.rows).toInt + 1, math.sqrt(weights.rows).toInt, index)
      val gcd: Int = lala(column.length)
      plot += image(new DenseMatrix[Double](gcd,  column.length / gcd, column.toArray), offset = (-extreme, extreme))
      index += 1
    }
    f2.saveas("image.png")
  }

  def debugSample(value: DenseMatrix[Double]): Unit = {
    val f2 = Figure()
    val plot = f2.subplot(0)
    val length = value.rows * value.cols
    val gcd: Int = lala(length)
    plot += image(new DenseMatrix[Double](gcd, length / gcd, value.toArray))
    f2.saveas("image.png")
  }

  private def getHiddenProp(rbm: RestrictedBoltzmannMachine, visibleData: Array[(Byte, DenseMatrix[Double])]) =
    visibleData.map {
      case (label, image) => (label, rbm.calcVisibleStateToHiddenProbabilities(rbm.weights, rbm.hiddenBias, image))
    }

  def trainFirstLayer(trainingData: Array[(Byte, DenseMatrix[Double])]): RestrictedBoltzmannMachine = {
    train("first", trainingData, 1000, 200)
  }

  def trainStackedLayer(index: Int)(trainingData: Array[(Byte, DenseMatrix[Double])]): RestrictedBoltzmannMachine = {
    train(s"stacked$index", trainingData, 2500, 100)
  }

  def hotEncodeLabel(label: Byte): List[Double] = List.fill[Double](label)(0) ::: (1.0 :: List.fill[Double](10 - label - 1)(0))

  def trainFinalLayer(trainingData: Array[(Byte, DenseMatrix[Double])]): RestrictedBoltzmannMachine = {

    train("final", trainingData.map {
      case (label, matrix) =>
        (label, new DenseMatrix(10 + matrix.rows, 1, hotEncodeLabel(label).toArray ++ matrix.toArray))
    }, 2500, 100)
  }

  def loadOrTraing(args: Array[String], trainingData: Array[(Byte, DenseMatrix[Double])], index: Int = 1, trainF: Array[(Byte, DenseMatrix[Double])] => RestrictedBoltzmannMachine = trainFirstLayer): RestrictedBoltzmannMachine =
    if (args.length >= index) {
      val argsIndex = index - 1
      println("loading Model: ", args(argsIndex))
      loadModel(args(argsIndex)) match {
        case Left(e) => throw e
        case Right(x) => x
      }
    } else {
      trainF(trainingData)
    }

  def main(args: Array[String]): Unit = {

    val allNumberTrainindData = MNISTDataset.trainingData.toArray

    val trainingData2 = allNumberTrainindData.filter(_._1 == 2)
    val trainingData3 = allNumberTrainindData.filter(_._1 == 8)
    val trainingData = allNumberTrainindData.map(pair => (pair._1, imageToMatrix(pair._2)))

    val rbm = loadOrTraing(args, trainingData, 1, trainFirstLayer)
    val trainingDataRBM2 = getHiddenProp(rbm, trainingData)
    val rbm2 = loadOrTraing(args, trainingDataRBM2, 2, trainStackedLayer(1))
    val trainingDataRBM3 = getHiddenProp(rbm2, trainingDataRBM2)
    val rbm3 = loadOrTraing(args, trainingDataRBM3, 3, trainFinalLayer)

    debugTrainedWeights(rbm.weights, "First Layer")
    debugTrainedWeights(rbm2.weights, "Second Layer")
    debugTrainedWeights(rbm3.weights, "Final Layer")

    for (i <- 0 until 10) {

      val label = hotEncodeLabel(9).toArray

      def clampLabels(visible: DenseMatrix[Double]): DenseMatrix[Double]
      = {
        visible(0 until 10, 0) := new DenseVector(label)
        visible
      }

      val visibleProp3 = rbm3.generateSample(50000, clampLabels)

      val visibleProp2 = rbm2.calcHiddenStateToVisibleProbabilities(rbm2.weights, rbm2.visibleBias, visibleProp3(10 until 110, 0 to 0))
      val image = rbm.calcHiddenStateToVisibleProbabilities(rbm.weights, rbm.visibleBias, visibleProp2)

      debugSample(image)

    }

//    val image = rbm.generateSample(10000)
//    val image2_1 = rbm.generateSampleFromInput(trainingData3.head._2, 0)
//    val image2_2 = rbm.generateSampleFromInput(trainingData3.head._2, 1)
//    val image2 = rbm.generateSampleFromInput(trainingData3.head._2, 2)
//    val image3 = rbm.generateSampleFromInput(trainingData3.head._2, 10)
//    val image4 = rbm.generateSampleFromInput(trainingData3.head._2, 1000)
//
//    println("minW", rbm.weights(*, ::).map(min(_)).toArray.mkString(","))
//    println("maxW", rbm.weights(*, ::).map(max(_)).toArray.mkString(","))
//
//    debugSample(image)
//    debugSample(image2_1)
//    debugSample(image2_2)
//    debugSample(image2)
//    debugSample(image3)
//    debugSample(image4)

  }

  private def imageToMatrix(image: DenseMatrix[Double]): DenseMatrix[Double] =
    new DenseMatrix(28*28, 1, image.toArray)

  private def imagesToMatrix(images: Array[DenseMatrix[Double]]): DenseMatrix[Double] =
    new DenseMatrix(28*28, images.length, images.flatMap(_.toArray).toArray)

  def train(stepName: String, trainingData: Array[(Byte, DenseMatrix[Double])], n_iterations: Int, n_hidden: Int, learningRate: Double = 0.005): RestrictedBoltzmannMachine = {

    require(trainingData.head._2.cols == 1)

    val n_visible = trainingData.head._2.rows

    val rbm = new RestrictedBoltzmannMachine(n_visible, n_hidden)

    // debugTrainingData(trainingData)

    val trainingMiniBatches: Array[Array[(Byte, DenseMatrix[Double])]] = trainingData.grouped(100).toArray

    val momentumSpeed = new DenseMatrix[Double](rbm.weights.rows, rbm.weights.cols)
    val momentumSpeedVB = new DenseVector[Double](rbm.visibleBias.length)
    val momentumSpeedHB = new DenseVector[Double](rbm.hiddenBias.length)

    var i = 0
    while(i < n_iterations) {
      println(s"$stepName | Iteration $i")
      for {
        batch <- trainingMiniBatches
      } {
        val (_, images) = batch.unzip
        val input = new DenseMatrix[Double](n_visible, images.length, images.flatMap(_.toArray))
        val (gradWeightsApprox, gradVB, gradHB) = rbm.cd1(rbm.weights, rbm.visibleBias, rbm.hiddenBias, input)
        momentumSpeed := momentumSpeed * 0.9 +:+ gradWeightsApprox
        momentumSpeedVB := momentumSpeedVB * 0.9 +:+ gradVB
        momentumSpeedHB := momentumSpeedHB * 0.9 +:+ gradHB
        rbm.weights := rbm.weights +:+ (momentumSpeed * learningRate)
        rbm.visibleBias := rbm.visibleBias +:+ (momentumSpeedVB * learningRate)
        rbm.hiddenBias := rbm.hiddenBias +:+ (momentumSpeedHB * learningRate)
      }
      i += 1
    }

    storeModel(rbm, s"step=${stepName}_#it=${n_iterations}_#h=$n_hidden")

    rbm
  }

  def storeModel(restrictedBoltzmannMachine: RestrictedBoltzmannMachine, name: String): Unit = {
    val date = nowAsOrdableString
    val folderPath = s"/Users/benikm91/Documents/Programming/Private/Scala/RestrictedBoltzmannMaschine/models/$date"
    new java.io.File(folderPath).mkdirs
    val oos = new ObjectOutputStream(new FileOutputStream(s"$folderPath/$name"))
    oos.writeObject(restrictedBoltzmannMachine)
    oos.close()
  }

  def loadModel(name: String): Either[Throwable, RestrictedBoltzmannMachine] =
    catching(classOf[Throwable]) either {
      val ois = new ObjectInputStream(new FileInputStream(s"/Users/benikm91/Documents/Programming/Private/Scala/RestrictedBoltzmannMaschine/models/$name"))
      val rbm = ois.readObject.asInstanceOf[RestrictedBoltzmannMachine]
      ois.close()
      rbm
    }

  def nowAsOrdableString: String = new SimpleDateFormat("yyyy-MM-dd_HH:mm").format(new java.util.Date())

}
