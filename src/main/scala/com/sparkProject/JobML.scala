package com.sparkProject

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{TrainValidationSplit, CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

import org.apache.spark.ml.param
import org.apache.spark.ml.linalg.Vectors

/**
  * Created by errochdi on 27/10/16.
  */
object JobML {
  def main(args: Array[String]): Unit = {


    // SparkSession configuration
    val spark = SparkSession
      .builder
      .appName("spark session TP_parisTech")
      .getOrCreate()

    val sc = spark.sparkContext

    //Chargement du csv issus de la première partie
    /**    val df = spark
      .read
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .option("comment", "#")
      .csv("/home/errochdi/Cours_MS_BIG_DATA/Spark/cleanedDataFrame/clean_data.csv")
      *
      */

    //Utilisation de parquet
    val df = spark
      .read.parquet("/home/errochdi/Cours_MS_BIG_DATA/Spark/cleanedDataFrame.parquet")

    println("number of columns", df.columns.length)
    println("number of rows", df.count)

    /**
      * a) Mise en forme des colonnes
        La plupart des algorithmes de machine learning dans Spark requièrent que les colonnes utilisées en input du modèle (les features du modèle) soient regroupées dans une seule colonne qui contient des vecteurs.
        => Ne pas mettre les colonnes “koi_disposition” et “rowid” dans les features.
        => ATTENTION : la méthode spark permettant de construire la colonne Features ne marche pas pour les colonnes ne contenant que des “Strings”.
      */
     var analysisData  = df

    //La liste des colonnes
    val liste_colonnes = df.columns.filter(x => x != "rowid").filter(x => x != "koi_disposition")

    println(liste_colonnes);



    val assembler = new VectorAssembler()
      .setInputCols(liste_colonnes)
      .setOutputCol("features")

    val output = assembler.transform(df)
    val liste_col = output.columns

    for( column_name <- liste_col){
      println( "nom des colonnes: " + column_name );
      }

    output.select("rowid","koi_period","koi_disposition","features")show(5);

    /** b) Travailler avec les chaînes de caractères Un algorithme ne peut pas utiliser
       une chaîne de caractère pour faire des calculs, il faut donc convertir ces chaînes en valeurs numériques.
       La colonne des labels
       est une colonne de Strings (“CONFIRMED” ou “FALSE-POSITIVE”), qu’il faut convertir en une colonne de 0 et de 1, pour une classification binaire.
      */

    //re-codage de la colonne "koi_disposition"
    val indexer = new StringIndexer()
      .setInputCol("koi_disposition")
      .setOutputCol("labels")

    val df_Clean = indexer.fit(output).transform(output)
    //Verification
    df_Clean.select("koi_disposition","labels","features").show(10)


    /********************/
    /********************/
    /**Machine Learning**/
    /********************/
    /********************/

    /** a) Splitter les données en Training Set et Test Set
           Séparer les données aléatoirement en un training set (90% des données)
           qui servira à l’entraînement du modèle et un test set (10% des données)
           qui servira à tester la qualité du modèle sur des données que le modèle
           n’a jamais vu lors de son entraînement.
    **/

    val Array(trainingData, testingData) = df_Clean.randomSplit(Array(0.9, 0.1))

    println("Nombre lignes training: ",trainingData.count())
    println("Nombre lignes test: ",testingData.count())

    println("Donnees training")
    trainingData.select("features").show(5)
    println("Donnees test")
    testingData.select("features").show(5)

    /**
      * b) Entraînement du classifieur et réglage des hyper-paramètres de l’algorithme.
      * Le classifieur que nous allons utiliser repose sur une régression logistique:
      * http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.classification.LogisticRegression
      * (que vous verrez en cours cette année) avec une régularisation dans la fonction de coût qui permet de pénaliser
      * les features ayant peu d’impact sur la classification: c’est la méthode du LASSO (que vous verrez également en détails).
      */


    //Créer une grille de valeurs à tester pour les hyper-paramètres.

    //Pour la grille on veut tester des valeurs de régularisation de 10e-6 à 1 par pas de 0.5 en échelle logarithmique.
    val gridtmp = -6.0 to (0.0 , 0.5) toArray
    val grid_array = gridtmp.map(x => math.pow(10,x))

    val lr = new LogisticRegression()
                  .setElasticNetParam(1.0)  // L1-norm regularization : LASSO
                  .setLabelCol("labels")
                  .setStandardization(true)  // to scale each feature of the model
                  .setFitIntercept(true)  // we want an affine regression (with false, it is a linear regression)
                  .setTol(1.0e-5)  // stop criterion of the algorithm based on its convergence
                  .setMaxIter(300)  // a security stop criterion to avoid infinite loops

    val paramGrid = new ParamGridBuilder().addGrid(lr.regParam,grid_array).build()

    /** En chaque point de la grille séparer le training set en un ensemble de training (70%) et un ensemble de validation (30%).
      * Entraîner un modèle sur le training set et calculer l’erreur du modèle sur le validation set.
      */

    val evaluator = new  BinaryClassificationEvaluator().setLabelCol("labels")

    val TrainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)//
      .setEvaluator(evaluator)//
      .setEstimatorParamMaps(paramGrid)//Les paramètres initialisés
      .setTrainRatio(0.7) //ratio à appliquer

      // Exécution du modèle
    println("Runining the model ***********")
    val model = TrainValidationSplit.fit(trainingData)

    val DataFrameWithPredictions = model.transform(testingData)
    DataFrameWithPredictions.select("labels","features","prediction").groupBy("labels", "prediction").count.show()

    //Affichage du score
    println("Le score de la prédiction.....")
    evaluator.setRawPredictionCol("prediction")
    println(evaluator.evaluate(DataFrameWithPredictions))
    //Saugvegarde du modèle
    model.write.overwrite().save("/home/errochdi/Cours_MS_BIG_DATA/Spark/stars.model")


  }

}
