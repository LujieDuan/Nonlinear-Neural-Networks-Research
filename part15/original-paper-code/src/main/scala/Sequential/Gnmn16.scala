package Sequential

import java.io.{File, PrintWriter}

import Utils.{Compute, Control}

import scala.io.StdIn

/**
  * Created by LD on 2017-02-21.
  * General program for simulating the NLA: Network Learning Algorithm
  */
class Gnmn16 {

  /** Maximum number of iterations allowed */
  val MaxItr = 10000

  /** Maximum number of input-output pairs */
  val Max = 5000

  /** Error tolerance */
  val EP = 0.4

  val EPP = 0.0001

  /** Types */
  class Lnk1 {
    var cx: Int = 0 //x coordinate
    var cy: Int = 0 //y coordinate
    var cn: Double = 0 //constant
    var lc: Double = 0 //coeff lambda
    var le: Double = 0 //exponent lambda
    var y: Double = 0 //function computed at this level
    var dp: Double = 0 //differential to be propagated
    var lo: Lnk2 = null //input to this layer
    var hi: Lnk3 = null //higher layer
    var nx: Lnk1 = null //next neuron
    var pr: Lnk1 = null //previous neuron
    var dn: Lnk5 = null //dummy nodes
  }

  def newLnk1: Lnk1 = {
    new Lnk1
  }


  class Lnk2 {
    var cf: Double = 0 //coefficient
    var dcf: Double = 0 //derivative wrt coeff
    var nt: Lnk2 = null //next term
    var ni: Lnk4 = null //next input
  }

  def newLnk2: Lnk2 = {
    new Lnk2
  }

  class Lnk4 {
    var vl: Boolean = false //valid bit
    var ex: Double = 0 //exponent
    var dex: Double = 0 //derivative wrt exponent
    var ni: Lnk4 = null //next input
    var nd: Lnk1 = null //corresponding node
  }

  def newLnk4: Lnk4 = {
    new Lnk4
  }

  class Lnk3 {
    var nd: Lnk1 = null //higher layer
    var dp: Double = 0 //differential propagated
    var nx: Lnk3 = null //next node
  }

  def newLnk3: Lnk3 = {
    new Lnk3
  }

  class Lnk5 {
    var nd: Lnk1 = null //higher layer
    var nx: Lnk5 = null //next node
    var cf: Double = 0 //coefficient
  }

  def newLnk5: Lnk5 = {
    new Lnk5
  }

  def BigType = Array.ofDim[Double](3, 3)
  def VerySmlr = new Array[Double](3)
  def VerySmli = new Array[Int](3)
  def MedInt = new Array[Int](Max)
  def MedRl = new Array[Double](Max)
  def Limarr = new Array[Double](6)
  def Inparr = Array.ofDim[Double](Max, 6) //Up to 6 dimensions allowed

  /** Variables */
  var nonr = new Array[Int](16) //[0..15]
  var mini = Limarr
  var maxi = Limarr

  var seed: Int = 1
  var cur :Int = 0
  var num :Int = 0
  var count :Int = 0
  var counter :Int = 0
  var corr: Int = 0
  var lcorr: Int = 0
  var curdim: Int = 0
  var zercor: Int = 0
  var onecor: Int = 0
  var newaddcnt: Int = 0
  var checker :Int = 0
  var llcorr: Int = 0

  var dow :Double = 0
  var p: Double = 0
  var q :Double = 0           //Lambda
  var err :Double = 0
  var sec :Double = 0
  var prev :Double = 0
  var prod :Double = 0
  var r :Double = 0
  var s :Double = 0

  var ep3: Double = 0
  var epsilon: Double = 0

  var total = VerySmlr
  var dawp = VerySmlr
  var x = VerySmlr

  var d = MedRl              //Original Output?
  var l = MedRl              //Original X
  var m = MedRl              //Original Y
  var u = Inparr
  var inp = Inparr


  var tf = new StringBuilder  //Output Stream
  var tf2 = new StringBuilder
  var ff = new String         //Input File Name
  var dire = new String

  var bul: Boolean = false
  var one: Boolean = false
  var two: Boolean = false

  var count2 = VerySmli
  var nosn = VerySmli
  var t1 = newLnk1
  var opphigh = newLnk1
  var lowest = newLnk1
  var highest = newLnk1


  /**
    * Produce random values in the interval [0, 1]
    */
  def random: Double = {
    val modulus = 65536
    val multiplier = 25173
    val increment = 13849
    seed = (multiplier*seed+increment) % modulus
    seed.toDouble/modulus
  }

  /**
    * Creates the network according to specifications and initializes it
    */
  def crtnet: Unit = {

    println("How many layers?")
    val nolr = StdIn.readInt()
    nonr(0) = 0
    var nstart: Lnk1 = null
    var start: Lnk1 = null
    var t1: Lnk1 = null

    var last: Lnk1 = null
    var nxlo: Lnk1 = null
    var nxlo2: Lnk1 = null

    for (i <- 0 until nolr) {
      start = nstart
      println(s"How many neurons at layer $i?")
      nonr(i) = StdIn.readInt()
      for (j <- 0 until nonr(i)) {
        t1 = newLnk1
        t1.pr = null
        t1.nx = last

        if (last == null)
          opphigh = t1
        else
          last.pr = t1

        last = t1
        if (i == 0) {
          t1.cx = i + 1
          t1.cy = j + 1
          t1.dn = null
          t1.lo = null
          t1.hi = null
          t1.y = 0
          lowest = t1
        }
        else {
          println("Enter values of lambdas: ")
          t1.lc = StdIn.readDouble()
          t1.le = StdIn.readDouble()
          t1.cn = Compute.random2 - 0.5
          t1.cx = i + 1 //scale x since i start from 0
          t1.cy = j + 1 //scale y
          t1.hi = null
          t1.lo = null
          t1.y = 0
          nxlo2 = start

          while (nxlo2 != null) {
            val t5 = newLnk5
            t5.nd = nxlo2
            t5.nx = t1.dn
            t1.dn = t5
            t5.cf = 0
            println(s"Connection from ${nxlo2.cx}, ${nxlo2.cy} to ${t1.cx}, ${t1.cy}?")
            val resp = StdIn.readChar()
            if (resp == 'y' || resp == 'Y'){
              tf2.++=(s"(${nxlo2.cx}, ${nxlo2.cy}) --> (${t1.cx}, ${t1.cy})")
              val t2 = newLnk2
              t2.nt = t1.lo
              t1.lo = t2
              t2.cf = Compute.random2 - 0.5
              t2.dcf = 0
              t2.ni = null
              nxlo = start

              while (nxlo != null){
                val t4 = newLnk4
                t4.ni = t2.ni
                t2.ni = t4
                t4.dex = 0
                t4.nd = nxlo
                if (nxlo == nxlo2){
                  t4.ex = 1
                  t4.vl = true
                }
                else {
                  t4.ex = 0
                  t4.vl = false
                }
                nxlo = nxlo.nx
              }
            }
            val t3 = newLnk3
            t3.dp = 0
            t3.nd = t1
            t3.nx = nxlo2.hi
            nxlo2.hi = t3
            nxlo2 = nxlo2.nx
          }
        }
      }
      nstart = t1
    }
    highest = t1
  }

  /**
    * Computes the network's output for the current input
    */
  def sigmoid: Double = {

    var t1: Lnk1 = lowest.pr
    var temp: Double = 0
    while (t1 != null) {
      var t2 = t1.lo
      var sum = t1.cn
      var prod: Double = 0

      while (t2 != null){
        prod = t2.cf
        var t4 = t2.ni

        while (t4 != null) {
          if (t4.vl) prod = prod * Compute.pr(t4.nd.y, t4.ex)
          t4 = t4.ni
        }
        t4 = t2.ni
        while (t4 != null) {
          if (t4.vl){
            var t3 = t4.nd.hi

            while (t3.nd != t1) t3 = t3.nx
            t3.dp += prod / Compute.pr(t4.nd.y, t4.ex) * t4.ex * Compute.pr(t4.nd.y, t4.ex - 1)
          }
          t4 = t4.ni
        }
        sum += prod
        t2 = t2.nt
      }

      temp = 1.toDouble / (1.toDouble + scala.math.exp(-sum))
      t1.y = temp
      t1 = t1.pr
    }
    temp
  }


  /**
    * runs the forward pass and assigns the result to 'sec'
    */
  def fwd = {
    var t1 = highest
    while (t1 != lowest) {
      var t3 = t1.hi
      while (t3 != null){
        t3.dp = 0
        t3 = t3.nx
      }
      t1 = t1.nx
    }
    sec = sigmoid
  }

  /**
    * backward pass of the back propagation algorithm... Computes all derivatives with
    * respect to the error and modifies parameter values
    */
  def backtrack = {
    dow = (sec - d(cur))*sec*(1-sec)
    var t1 = highest
    var t2 = t1.lo
    var t5 = t1.dn
    var prod: Double = 0
    var prevsum: Double = 0

    while(t5 != null) {
      t5.cf -= t1.lc * dow * t5.nd.y
      t5 = t5.nx
    }

    if(t1.lo != null) {
      t1.cn -= t1.lc * dow
    }

    while (t2 != null){
      prod = 1
      var t4 = t2.ni
      while(t4 != null) {
        if(t4.vl) prod = prod * Compute.pr(t4.nd.y, t4.ex)
        t4 = t4.ni
      }
      t4 = t2.ni
      while(t4 != null) {
        t4.dex = dow * prod * t2.cf * Compute.lan(t4.nd.y)
        if((t4.ex - t1.le * t4.dex) >= 0 || !t4.vl)
          t4.ex -= t1.le * t4.dex
        else
          t4.ex = 0
        t4 = t4.ni
      }
      t2.dcf = dow * prod
      t2.cf -= t1.lc * t2.dcf
      t2 = t2.nt
    }

    t1.dp = dow
    t1 = t1.nx

    while (t1 != null){
      var t3 = t1.hi
      while (t3 != null){
        var expdiff: Double = 0
        prevsum += t3.nd.dp * t3.dp
        t3 = t3.nx
      }
      t5 = t1.dn
      while(t5 != null) {
        t5.cf -= t1.lc * prevsum * t5.nd.y * (1 - t1.y) * t1.y
        t5 = t5.nx
      }

      if(t1.lo != null) {
        t1.cn -= t1.lc * prevsum * (1 - t1.y) * t1.y
      }

      t2 = t1.lo

      while(t2 != null) {
        prod = 1
        var t4 = t2.ni
        while(t4 != null) {
          if(t4.vl) prod = prod * Compute.pr(t4.nd.y, t4.ex)
          t4 = t4.ni
        }
        t4 = t2.ni
        while(t4 != null) {
          t4.dex = prevsum * (1 - t1.y) * t1.y * t2.cf * prod * Compute.lan(t4.nd.y)
          if((t4.ex - t1.le * t4.dex) >= 0 || !t4.vl)
            t4.ex -= t1.le * t4.dex
          else
            t4.ex = 0
          t4 = t4.ni
        }
        t2.dcf = prevsum * prod * (1 - t1.y) * t1.y
        t2.cf -= t1.lc * t2.dcf
        t2 = t2.nt
      }
      t1.dp = prevsum * (1 - t1.y) * t1.y
      t1 = t1.nx
    }
  }

  /**
    * retracts unnecessary inputs and synacts
    */
  def remifnec = {
    var t1 = highest

    while(t1 != lowest) {
      var already = false
      var t2 = t1.lo
      var first = t2
      if(t1 == null) {
        already = true
      }

      while(t1 != null && scala.math.abs(t2.cf) < (0.01 * t1.lc)) {
        first = t2.nt
        print(0)
        t2 = t2.nt
      }
      var last = first

      while(t2 != null) {
        if(math.abs(t2.cf) < (0.01 * t1.lc)) {
          last.nt = t2.nt
          print(0)
        }
        else {
          var t4 = t2.ni
          while(t4 != null) {
            if(t4.vl && t4.ex < (0.01 * t1.le)){ //?????
              t4.vl = false
              print("-")
              t4.ex = 0
            }
            t4 = t4.ni
          }
          last = t2
        }
        t2 = t2.nt
      }
      t1.lo = first
      t1 = t1.nx
    }
  }

  /**
    * Adds new synacts or inputs if required
    */
  def addinfec = {
    var t1 = highest
    while(t1 != lowest) {
      var t5 = t1.dn
      while(t5 != null) {
        if(math.abs(t5.cf) > (5 * t1.lc)) {
          t5.cf = 0
          print(1)
          var t2n = newLnk2
          t2n.nt = t1.lo
          t2n.ni = null
          var t2 = t1.lo //???
          t2n.cf = 0
          t1.lo = t2n
          var t5n = t1.dn

          while(t5n != null) {
            var t4n = newLnk4
            t4n.dex = 0
            t4n.nd = t5n.nd
            t4n.ni = t2n.ni
            if(t4n.nd == t5.nd) {
              t4n.ex = 1
              t4n.vl = true
            }
            else {
              t4n.ex = 0
              t4n.vl = false
            }
            t2n.ni = t4n
            t5n = t5n.nx
          }
        }
        t5 = t5.nx
      }
      var t2 = t1.lo
      while(t2 != null) {
        var t4 = t2.ni
        while(t4 != null) {
          if(t4.ex > 5 * t1.le && !t4.vl) {
            print('.')
            t4.vl = true
            t4.ex = 0.01 * t1.le
          }
          t4 = t4.ni
        }
        t2 = t2.nt
      }
      t1 = t1.nx
    }
  }

  /**
    * returns the value of error
    */
  def error: Double = {
    prev = err
    err = scala.math.abs(d(cur) - sec)
    err
  }

  /**
    * Assigns true to 'one' if error less than the less stringent condition...elsem assigns it false.
    * Assigns true to 'two' if error less than the more stringent condition...else, assigns it false.
    */
  def correct = {
    one = false
    two = false
    if (error <= EP + (num - lcorr).toDouble / num * 0.05)
      one = true
    if (error <= EP)
      two = true
  }


  /**
    * writes the learnt functions to a file
    */
  def wrtres = {
    var t1 = highest

    while (t1 != lowest) {
      tf.++=("------------------------------------\n\n")
      tf.++=(s"Layer ${t1.cx} Neuron ${t1.cy}\n\n")
      tf.++=("------------------------------------\n\n")
      tf.++=(s"Values of Lambdas: ${t1.lc} ${t1.le}\n\n")
      tf.++=(s"Threshold ${-t1.cn}\n\n")
      var t2 = t1.lo
      var i = 0

      if(t1 == null) tf.++=("No synapses\n")

      while (t2 != null){
        tf.++=("------------------------------------\n\n")
        tf.++=(s"Synapse \n\n")
        tf.++=("------------------------------------\n\n")
        tf.++=(s"Coeff ${t2.cf}\n")
        tf.++=("-----------------\n")

        var t4 = t2.ni
        while (t4 != null){
          if(t4.vl) tf.++=(f"${t4.ex}%6s") else tf.++=("      ")
          t4 = t4.ni
        }
        tf.++=("\n\n")
        t2 = t2.nt
        i += 1
      }
      t1 = t1.nx
    }
  }

  /**
    * runs iterations until either all the input-output pairs have been learnt or the highest
    * number of iterations has been reached
    */
  def all = {
    var prevchk = 0
    cur = 0
    curdim = 0
    lowest.y = u(cur)(curdim)
    var t1 = lowest.nx

    while (t1 != null){
      curdim += 1
      t1.y = u(cur)(curdim)
      t1 = t1.nx
    }

    fwd

    bul = false
    corr = 0
    lcorr = 0

    do {
      newaddcnt += 1
      curdim = 1
      lowest.y = u(cur)(curdim)
      t1 = lowest.nx

      while (t1 != null){
        curdim += 1
        t1.y = u(cur)(curdim)
        t1 = t1.nx
      }

      fwd
      correct

      if (!two){
        backtrack
        if (!one) checker = 0

        cur = (cur % num) + 1

        if (one) {
          corr += 1
          checker += 1
        }
      }
      else {
        corr += 1
        checker += 1
        cur = (cur % num) + 1
      }
      if (cur == num) {
        if (corr != lcorr){
          println(s"$corr   $prevchk   ${count.toDouble / num + 0.5}")
          lcorr = corr
        }
        if(count > num * 10) remifnec
        addinfec
        corr = 0
      }

      fwd
      count += 1

      if (checker > prevchk){
        println(checker)
        prevchk += 1
      }
    } while (checker < num && count <= num * MaxItr)
  }


  /**
    * reads the value of 'seed' to be used for the random number generator from a file
    */
  def globini = {
    Control.using(io.Source.fromFile("seed")) { source => {
      for (line <- source.getLines) {
        seed = line.mkString.toInt
      }
    }}
  }

  /**
    * Calls 'wrtres' to write the learnt parameters to the output file.
    * Output the number of iterations required if a solution was found and informs if number
    * of iterations exceeded the limit
    */
  def respond = {
    if (count > num * MaxItr) {
      wrtres
      tf.++=("Number of iterations exceeded!\n")
    }
    else if (checker >= num) {
      wrtres
      tf.++=("\n")
      tf.++=(s"Got it!... In ${count/num} iterations\n")
      tf.++=("\n")
      tf.++=(s"Values of lambdas = $q $r\n")
      tf.++=("\n")
    }
  }

  /**
    * reads input-output pairs from the input file
    */
  def readinp = {
    var iterator = 0
    val dim = nonr(0)
    Control.using(io.Source.fromFile(s"src/main/resources/sample-$ff" )) { source => {
      val inputIntegerList = source.mkString.split(" ").map(_.toInt)
      num = inputIntegerList(iterator)
      iterator += 1
      assert(num <= Max)

      for (i <- 0 until dim){
        inp(0)(i) = inputIntegerList(iterator)
        iterator += 1
        mini(i) = inp(0)(i)
        maxi(i) = inp(0)(i)
      }

      d(0) = inputIntegerList(iterator)
      iterator += 1

      for (i <- 1 until num){
        for (j <- 0 until dim){
          inp(i)(j) = inputIntegerList(iterator)
          iterator += 1
          if (inp(i)(j) < mini(j)) mini(j) = inp(i)(j) else if (inp(i)(j) > maxi(j)) maxi(j) = inp(i)(j)
        }
        d(i) = inputIntegerList(iterator)
        iterator += 1
      }

      for (j <- 0 until dim){
        maxi(j) += 1
        mini(j) -= 1
      }
    }}

    for (i <- 0 until num; j <- 0 until dim)
      u(i)(j) = (inp(i)(j) - mini(j)) / (maxi(j) - mini(j))
  }

  /**
    * writes generalization figures to the output file
    */
  def genfigr = {
    val fff = ff
    ff = "complete-" + fff
    readinp
    zercor = 0
    onecor = 0

    for (i <- 0 until num){
      cur = i
      curdim = 0
      lowest.y = u(cur)(curdim)
      t1 = lowest.nx

      while (t1 != null){
        curdim += 1
        t1.y = u(cur)(curdim)
        t1 = t1.nx
      }
      fwd
      correct

      if (one && d(cur) == 0) zercor += 1
      if (one && d(cur) == 1) onecor += 1
    }

    tf.++=(s"Getting correct 0 values for $zercor input values\n")
    tf.++=(s"Getting correct 1 values for $onecor input values\n")


    val pw = new PrintWriter(new File(s"src/main/resources/gnmn16-result-$fff" ))
    pw.write(tf.toString())
    pw.close()
    println(tf.toString())
    showInput
    println(s"Write Result to file src/main/resources/gnmn16-result-$fff")
  }

  /**
    * saves value of seed
    */
  def saveseed = {
    val pw = new PrintWriter(new File("seed" ))
    pw.write(seed)
    pw.close()
  }

  /**
    * main program
    */
  def MainRouting = {
    //globini // loads seed value for 'random'
    crtnet // create the network
    println("Enter the data file number:")
    ff = StdIn.readLine()
    readinp // reads input-output pairs
    val start = System.currentTimeMillis
    all //runs iterations
    respond // outputs learnt functions
    val totalTime = System.currentTimeMillis - start
    genfigr // outputs generalization figures
    //saveseed // saves seed value
    println("Elapsed time: %1d ms".format(totalTime))
  }

  /**
    * Show input in matrix form
    */
  def showInput: Unit = {
    /*    val inputArray = Array.fill[Int](maxl.toInt - minl.toInt + 1, maxm.toInt - minm.toInt + 1)(-1)
        for (i <- 0 until num){
          inputArray(l(i).toInt - 1)(m(i).toInt - 1) = d(i)
        }
        inputArray foreach { row => row foreach {element => if (element < 0) print(" ") else print(element)}; println }*/
  }


}
