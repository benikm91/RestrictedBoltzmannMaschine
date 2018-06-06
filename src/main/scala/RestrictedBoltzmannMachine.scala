package main.scala

import breeze.linalg.{*, DenseMatrix, DenseVector, max, min, sum}
import breeze.numerics.sigmoid

import scala.util.Random

@SerialVersionUID(123L)
class RestrictedBoltzmannMachine(n_visible: Int, n_hidden: Int) extends Serializable {

  val visibleBias: DenseVector[Double] = DenseVector.zeros[Double](n_visible)
  val hiddenBias: DenseVector[Double] = DenseVector.zeros[Double](n_hidden)
  val weights: DenseMatrix[Double] = DenseMatrix.rand[Double](n_hidden, n_visible).map(x => (x - 0.5) * .2)

  def calcHiddenStateToVisibleProbabilities(weights: DenseMatrix[Double], visibleBias: DenseVector[Double], hidden: DenseMatrix[Double]): DenseMatrix[Double]
    = {
    val res = weights.t * hidden
    sigmoid(res(::, *).map(_ +:+ visibleBias))
  }
  def calcVisibleStateToHiddenProbabilities(weights: DenseMatrix[Double], hiddenBias: DenseVector[Double], visible: DenseMatrix[Double]): DenseMatrix[Double]
    = {
    val res: DenseMatrix[Double] = weights * visible
    sigmoid(res(::, *).map(_ +:+ hiddenBias))
  }

  def configurationGoodness(weights: DenseMatrix[Double], visibleBias: DenseVector[Double], hiddenBias: DenseVector[Double], visible: DenseMatrix[Double], hidden: DenseMatrix[Double]) : Double
    = sum((hidden * visible.t) *:* weights) / visible.rows + sum(visibleBias.t * visible) + sum(hiddenBias.t * hidden)

  def configurationGoodnessGradient[T](visible: DenseMatrix[Double], hidden: DenseMatrix[Double]): DenseMatrix[Double]
    = (hidden * visible.t).map(_ / visible.rows)

  def sampleBernoulli(m: DenseMatrix[Double])(implicit r: Random): DenseMatrix[Int]
    = m.map(x => if (new breeze.stats.distributions.Bernoulli(x).draw()) 1 else 0)

  def cd1(weights: DenseMatrix[Double], visibleBias: DenseVector[Double], hiddenBias: DenseVector[Double], miniBatchVisible: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double], DenseVector[Double]) = {
    implicit val r: Random = new Random()
    val visibleProb = miniBatchVisible
    val visible = sampleBernoulli(miniBatchVisible)
    val hiddenProb = calcVisibleStateToHiddenProbabilities(weights, hiddenBias, visible.map(_.toDouble))
    val hidden = sampleBernoulli(hiddenProb)
    val reconstructionVisibleProb = calcHiddenStateToVisibleProbabilities(weights, visibleBias, hidden.map(_.toDouble))
    val reconstructionVisible = sampleBernoulli(reconstructionVisibleProb)
    val reconstructionHiddenProb = calcVisibleStateToHiddenProbabilities(weights, hiddenBias, reconstructionVisible.map(_.toDouble))
    val d_visibleBias = visibleProb -:- reconstructionVisibleProb
    val d_hiddenBias = hiddenProb -:- reconstructionHiddenProb
    val d_weights = configurationGoodnessGradient(visible.map(_.toDouble), hidden.map(_.toDouble)) -:- configurationGoodnessGradient(reconstructionVisible.map(_.toDouble), reconstructionHiddenProb)
    (d_weights, d_visibleBias(*, ::).map(x => sum(x)), d_hiddenBias(*, ::).map(x => sum(x)))
  }

  def generateSample(n_iterations: Int = 100, updateVisible: DenseMatrix[Double] => DenseMatrix[Double] = identity[DenseMatrix[Double]]): DenseMatrix[Double] = {
    implicit val r: Random = new Random()
    var i = 0
    var visible = updateVisible(DenseMatrix.rand[Double](weights.cols, 1))
    var hidden = DenseMatrix.rand[Double](weights.rows, 1)
    while (i < n_iterations) {
      visible = calcHiddenStateToVisibleProbabilities(this.weights, this.visibleBias, sampleBernoulli(hidden).map(_.toDouble))
      visible = updateVisible(visible)
      hidden = calcVisibleStateToHiddenProbabilities(this.weights, this.hiddenBias, sampleBernoulli(visible).map(_.toDouble))
      i += 1
    }
    visible
  }

  def generateSampleFromInput(p_visible: DenseMatrix[Double], n_iterations: Int = 1): DenseMatrix[Double] = {
    implicit val r: Random = new Random()
    var i = 0
    var visible = new DenseMatrix[Double](p_visible.rows * p_visible.cols, 1, p_visible.toArray)
    var hidden = DenseMatrix.rand[Double](weights.rows, 1)
    while (i < n_iterations) {
      hidden = calcVisibleStateToHiddenProbabilities(this.weights, this.hiddenBias, sampleBernoulli(visible).map(_.toDouble))
      visible = calcHiddenStateToVisibleProbabilities(this.weights, this.visibleBias, sampleBernoulli(hidden).map(_.toDouble))
      i += 1
    }
    visible
  }

}
