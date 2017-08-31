import breeze.linalg.DenseVector

object MnistLoader {

  /**
    * Load train/validation/testSet sets, with default rate of train:validation = 90:10
    * @return (train, validation, testSet)
    */
  def load(): DataSet = {
    val result = Seq[(DenseVector[Double], DenseVector[Double])]((DenseVector(0, 0), DenseVector(0, 1)),
      (DenseVector(0, 1), DenseVector(1, 0)),
      (DenseVector(1, 0), DenseVector(1, 0)),
      (DenseVector(1, 1), DenseVector(0, 1)))
    new DataSet(result, result, result)
  }
}
