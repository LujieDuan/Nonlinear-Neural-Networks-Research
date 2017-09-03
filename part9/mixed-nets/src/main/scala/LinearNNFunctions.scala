
import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by LD on 2017-05-17.
  * Translate from Python: https://github.com/mnielsen/neural-networks-and-deep-learning
  */
object LinearNNFunctions {

  /**
    *
    * @param z
    * @param t
    * @return
    */
  def LinearTransposeAndMultiply(z: DenseVector[Double], t: DenseVector[Double]): DenseMatrix[Double] = {
    DenseMatrix.tabulate(t.length, z.length){case (i, j) => t(i) * z(j)}
  }
}
