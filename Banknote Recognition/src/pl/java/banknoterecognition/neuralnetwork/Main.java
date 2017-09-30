package pl.java.banknoterecognition.neuralnetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import javax.management.modelmbean.DescriptorSupport;
import javax.print.attribute.standard.NumberOfDocuments;

import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.highgui.Highgui;

import java.awt.HeadlessException;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class Main {

	private static int numInputs = 64; // liczba wejsc sieci
	private static int numHidden = 128; // liczba neuronow w warstwie ukrytej
	private static int numOutput = 5;  // liczba wyjsc sieci

	private static double alpha = 0.5; // wspolczynnik uczenia (uzywany w alg wstecznej propagacji)
	private static int trainingCycles = 50;

	private static double weightsInpToHid[][] = new double[numInputs + 1][numHidden]; // wagi przychodzace do ukrytej warstwy
	private static double weightsHidToOut[][] = new double[numHidden + 1][numOutput]; // wagi wychodzace z ukrytej warstwy

	private static double[] inputs = new double[numInputs]; // wejsca sieci
	private static double[] hidden = new double[numHidden]; // neurony wartstwy ukrytej
	private static double[] expected = new double[numOutput]; // oczekiwane wyjscie
	private static double[] predicted = new double[numOutput];// przewidywane wyjscie (aktualnie obliczone)

	private static double[] errForOut = new double[numOutput]; // blad wyjscia
	private static double[] errForHid = new double[numHidden]; // blad warstwy ukrytej
	
	private static double errorOfEpoch = 0.0;
	private static double[] errorOfEpochPart = new double[numOutput];

	private static double trainOutput[][] = new double[][] { { 0.9, 0.1, 0.1, 0.1, 0.1 }, { 0.1, 0.9, 0.1, 0.1, 0.1 }, { 0.1, 0.1, 0.9, 0.1, 0.1 },
			{ 0.1, 0.1, 0.1, 0.9, 0.1 }, { 0.1, 0.1, 0.1, 0.1, 0.9 } };
			
	private static String nominals[] = {"10", "20", "50", "100", "200"};
	private static int numTrainImgForNom = 7;
	private static int numTestImgForNom = 3;

	private static File lib = new File("library//" + System.mapLibraryName("opencv_java2411"));
	
	private static String kindOfData;
	private static int minNumDescriptors;
	private static PrintWriter wyniki; 
	private static int numberOfApperances[] = {0, 0, 0, 0, 0};
	private static List<Double> correctInPerc = new ArrayList<>(); 

	
	public static void main(String[] args) throws FileNotFoundException {
		System.load(lib.getAbsolutePath());
		
		
		Scanner odczyt = new Scanner(System.in); //obiekt do odebrania danych od uzytkownika
		 
		System.out.println("Podaj ilosc neuronow w wartwie ukrytej");
	    numHidden = Integer.parseInt(odczyt.nextLine());
	    
	    System.out.println("Podaj ilosc wspolczynnik uczenia");
	    alpha = Double.parseDouble(odczyt.nextLine());
	    
	    System.out.println("Podaj liczbe epok");
	    trainingCycles = Integer.parseInt(odczyt.nextLine());
	    
	    //wyniki = new PrintWriter("numInputs_"+numInputs+""+"numHidden"+numHidden+""+"numOutput"+numOutput+""+"trainingCycles"+trainingCycles+".txt");
	    
	 
	      //System.out.println("Witaj "+imie); //wyswietlamy powitanie
	
		int numberOfTrainingImages = numTrainImgForNom * nominals.length;
		
		//lista nominalow (elementem jest lista deskryptorow z odpowiadajacymi oczekiwanymi wyjsciami
		List<List<InputOutputPair>> nominalList = new ArrayList<>();
		//liczba descriptorow w danym nominale
		int[] numOfDescInNominal = new int[nominals.length];
		//wyjscia oczekiwane dla wszystkich deskryptorow nominalu
		List<double[]> output = new ArrayList<double[]>();
		MatOfKeyPoint descriptors = new MatOfKeyPoint();
		
		
		kindOfData = "TRENINGOWE";
		/*DANE TRENINGOWE*/
		/*pobranie wszystkich deskryptorow wszystkich obrazkow*/
		for (int i = 0; i < nominals.length; i++) { //dla kazdego nominalu
			double sum = 0;
			List<InputOutputPair> nominal = new ArrayList<>();
			for (int j = 0; j < numTrainImgForNom; j++) { //przejdz przez wszystkie obrazki
				descriptors = getDataForImage(kindOfData, nominals[i], j + 1); //pobierz deskryptory jednego obrazka
				for(int k = 0; k < descriptors.size().height ; k++) {
					output.add(trainOutput[i]);
				}
				for(int l = 0; l < descriptors.size().height ; l++) {
					double[] desc = new double[64];
					for(int m = 0; m < descriptors.size().width ; m++) {
						desc[m] = descriptors.get(j, m)[0];
					}
					nominal.add(new InputOutputPair(desc, output.get(j)));
				}
				output.clear();
				sum += descriptors.size().height; 
				//wyniki.print(descriptors.size().height + "  + ");
				//System.out.print(descriptors.size().height + "  + ");
			}
			//wyniki.println("   =  " + sum);
			//System.out.println("   =  " + sum);
			nominalList.add(nominal);
			numOfDescInNominal[i] = (int)sum;
		}
		//wyniki.println();
		System.out.println();
		
		/* normalizacja wejsc (wymieszanie deskryptorow nominalow i zmniejszenie ich ilosci do ilosci deskryptorow nominalu, ktory ma ich najmniej */
		/* oraz przepisanie wszystkich deskryptorow do jednej listy w celu pozniejszego ich wymieszania i uzycia jako danych treningowych */
		List<InputOutputPair> allInOutPair = new ArrayList<InputOutputPair>();
		int index = indexOfMinVec(numOfDescInNominal);
		minNumDescriptors = numOfDescInNominal[index];
		for(int i = 0 ; i < nominals.length ; i++) {
			Collections.shuffle(nominalList.get(i));
			nominalList.set(i, nominalList.get(i).subList(0, minNumDescriptors - 1));
			allInOutPair.addAll(nominalList.get(i));
		}
		
		//wyniki.println("Min: " + numOfDescInNominal[index]);		
		//System.out.println("Min: " + numOfDescInNominal[index]);	
		
		Collections.shuffle(allInOutPair);
		
		
		
		/*DANE TESTOWE*/
		List<List<MatOfKeyPoint>> nominalsTest = new ArrayList<>();
		List<MatOfKeyPoint> imagesTest = new ArrayList<>();
		kindOfData = "TESTOWE";		
		for (int i = 0; i < nominals.length; i++) { //dla kazdego nominalu
			for (int j = 0; j < numTestImgForNom; j++) { //przejdz przez wszystkie obrazki
				imagesTest.add(getDataForImage(kindOfData, nominals[i], j + 1)); //pobierz deskryptory punktow obrazka
				//int result = testNetwork(descriptorsOfImage); // testuj siec (jeden obrazek)
				//wyniki.print("Wejscie: " + nominals[i] + " zl, zdj: " + (j + 1) + "   Wyjscie: " + nominals[result]);
			}
			nominalsTest.add(imagesTest);
		}
		
		
		
		
		/*TRENING*/
		initializeRandWeights();
		for (int epoch = 0; epoch < trainingCycles; epoch++) {
				for(InputOutputPair inputOutputPair: allInOutPair) {
					for(int i = 0 ; i < inputOutputPair.getDescriptorInput().length ; i++) {
						inputs[i] = inputOutputPair.getDescriptorInput()[i];
						//wyniki.format(" %.4f \t", inputs[i]);
					}
					//wyniki.println();
					for (int j = 0; j < numOutput; j++) { 
						expected[j] = inputOutputPair.getOutput()[j];
					}
					//for(int s = 0 ; s < expected.length ; s++)
					//	wyniki.print(expected[s] +" ");
					//wyniki.println();
					doFeedForward();
					doBackPropagation();
				}
				double sum = 0.0;
				for(int i = 0 ; i < errorOfEpochPart.length ; i++) {
					errorOfEpoch += errorOfEpochPart[i];
					errorOfEpochPart[i] = 0.0;
				}
				errorOfEpoch *= 0.5;
				//wyniki.println(errorOfEpoch);
				if(epoch == trainingCycles - 1) {
					System.out.print("Blad " + trainingCycles + " epoki: ");
					System.out.format("%.4f \n\n\n\n", errorOfEpoch);
				}
				errorOfEpoch = 0.0;
		}
		

		test(nominalsTest);
		//wyniki.close();
	}
	
	
	public static void test(List<List<MatOfKeyPoint>> nominalsTest) {
		kindOfData = "TESTOWE";		
		for (int i = 0; i < nominals.length; i++) { //dla kazdego nominalu
			for (int j = 0; j < numTestImgForNom; j++) { //przejdz przez wszystkie obrazki
				MatOfKeyPoint descriptorsOfImage = nominalsTest.get(i).get(j); //pobierz deskryptory punktow obrazka
				int result = testNetwork(descriptorsOfImage, i); // testuj siec (jeden obrazek)
				//wyniki.print("Wejscie: " + nominals[i] + " zl, zdj: " + (j + 1) + "   Wyjscie: " + nominals[result]);
				System.out.println("Wejscie: " + nominals[i] + " zl, zdj: " + (j + 1) + "   Wyjscie: " + nominals[result]);
			}
		} 
		
		double sum = 0.0;
		for(double percent : correctInPerc) {
			sum += percent;
		}
		double averageCorrPerc = sum / correctInPerc.size();
		//wyniki.println();
		//wyniki.println("Dobre wyniki dla : " + averageCorrPerc + " % deskryptorow");
		System.out.println();
		System.out.println("ilosc poprawnie zidentyfikowanych deskryptorow w zdjeciu: " + averageCorrPerc + " %");
	}

	//pobiera deskryptory punktow kluczowych dla danego obrazka banknotu
	private static MatOfKeyPoint getDataForImage(String kindOfData, String nominal, int imageNumber) {

		String object = "images//" + kindOfData + "//" + nominal + "//" + imageNumber + ".jpg";

		Mat objectImage = Highgui.imread(object, Highgui.CV_LOAD_IMAGE_COLOR);

		MatOfKeyPoint objectKeyPoints = new MatOfKeyPoint();
		FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.SURF);
		featureDetector.detect(objectImage, objectKeyPoints);

		List<KeyPoint> keypoints = objectKeyPoints.toList();
		
		MatOfKeyPoint objectDescriptors = new MatOfKeyPoint();
		DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
		descriptorExtractor.compute(objectImage, objectKeyPoints, objectDescriptors);
		
		
		String imagesWithKeypoints = "images//" + kindOfData + "//" + nominal + "//keypoints//" + imageNumber + ".jpg";
		// Create the matrix for output image.
        Mat outputImage = new Mat(objectImage.rows(), objectImage.cols(), Highgui.CV_LOAD_IMAGE_COLOR);
        Scalar newKeypointColor = new Scalar(255, 0, 0);

        
        Collections.shuffle(keypoints);
        //keypoints = keypoints.subList(0, 49);
        objectKeyPoints.fromList(keypoints);
        Features2d.drawKeypoints(objectImage, objectKeyPoints, outputImage, newKeypointColor, 0);
        Highgui.imwrite(imagesWithKeypoints, outputImage);
		
		

		return objectDescriptors;
	}

	// testuje siec dla jednego zdjecia banknotu (zestaw deskryptorow) (zwraca numer banknotu)
	private static int testNetwork(MatOfKeyPoint descriptors, int nominal) {
		for(int elem: numberOfApperances) {
			elem = 0;
		}
	 
		for(int i = 0 ; i < descriptors.size().height ; i++) {
			for(int j = 0 ; j < descriptors.size().width ; j++) {
				inputs[j] = descriptors.get(i, j)[0]; //kazdy deskryptor punktu wchodzi do SN
			}
			doFeedForward();
			
			int index = indexOfMaxVec(predicted);
			numberOfApperances[index]++;
		}
		//wyniki.println();
		double sumOfApperance = 0;
		for(int i: numberOfApperances) {
 			//wyniki.print("  " + i);
 			sumOfApperance += i;
		}
		double percentOfCorrAns = (double)(numberOfApperances[nominal] / sumOfApperance) * 100;
		correctInPerc.add(percentOfCorrAns);
		//wyniki.print("     ilosc trafnych odpowiedzi: " + percentOfCorrAns + "%");
		//wyniki.print("%");
		//wyniki.println();
		return indexOfMaxVec(numberOfApperances);
	}

	private static void initializeRandWeights() {
		for (int i = 0; i <= numInputs; i++) {
			for (int j = 0; j < numHidden; j++) {
				weightsInpToHid[i][j] = (new Random().nextDouble()) - 0.5;
			}
		}

		for (int i = 0; i <= numHidden; i++) {
			for (int j = 0; j < numOutput; j++) {
				weightsHidToOut[i][j] = (new Random().nextDouble()) - 0.5;
			}
		}
	}

	
	private static void doFeedForward() {
		double sum = 0.0;
		for (int i = 0; i < numHidden; i++) {
			sum = 0.0;
			for (int j = 0; j < numInputs; j++) {
				sum += weightsInpToHid[j][i] * inputs[j];
			}
			sum += weightsInpToHid[numInputs][i];
			hidden[i] = sigmoid(sum);
		}

		for (int i = 0; i < numOutput; i++) {
			sum = 0.0;
			for (int j = 0; j < numHidden; j++) {
				sum += hidden[j] * weightsHidToOut[j][i];
			}
			sum += weightsHidToOut[numHidden][i];
			predicted[i] = sigmoid(sum);
			//wyniki.print(predicted[i] + " ");
		}
		//wyniki.println();
	}


	//wykonuje algorytm wstecznej propagacji (uczenia na bledach). Dopasowujemy wagi zaleznie od roznicy miedzy obliczana a oczekiwana wartoscia
	private static void doBackPropagation() {
		for (int i = 0; i < numOutput; i++) {
			errorOfEpochPart[i] += Math.pow((expected[i] - predicted[i]), 2);
			errForOut[i] = (expected[i] - predicted[i]) * predicted[i] * (1.0 - predicted[i]);
		}

		for (int i = 0; i < numHidden; i++) {
			errForHid[i] = 0.0;
			for (int j = 0; j < numOutput; j++) {  
				errForHid[i] += errForOut[j] * weightsHidToOut[i][j];
			}
			errForHid[i] *= hidden[i] * (1.0 - hidden[i]);
		}

		for (int i = 0; i < numOutput; i++) {
			for (int j = 0; j < numHidden; j++) {
				weightsHidToOut[j][i] += (alpha * errForOut[i] * hidden[j]);
			}
			weightsHidToOut[numHidden][i] += (alpha * errForOut[i]);
		}

		for (int i = 0; i < numHidden; i++) {
			for (int j = 0; j < numInputs; j++) {
				weightsInpToHid[j][i] += (alpha * errForHid[i] * inputs[j]);
			}
			weightsInpToHid[numInputs][i] += (alpha * errForHid[i]);
		}

	}

	// oblicza funkcje sigmoid
	private static double sigmoid(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}

	

	private static int indexOfMaxVec(final double[] vector) { // dla zmiennoprzecinkowych
		// zwraca indeks najwiekszego elementu wektora
		int tmp = 0;
		double max = vector[tmp];

		for (int index = 0; index < numOutput; index++) {
			if (vector[index] > max) {
				max = vector[index];
				tmp = index;
			}
		}
		return tmp;
	}
	
	private static int indexOfMaxVec(final int[] vector) { // dla calkowitych
		// zwraca indeks najwiekszego elementu wektora
		int tmp = 0;
		double max = vector[tmp];

		for (int index = 0; index < numOutput; index++) {
			if (vector[index] > max) {
				max = vector[index];
				tmp = index;
			}
		}
		return tmp;
	}
	
	private static int indexOfMinVec(final int[] vector) { // dla calkowitych
		// zwraca indeks najwiekszego elementu wektora
		int tmp = 0;
		double min = vector[tmp];

		for (int index = 0; index < numOutput; index++) {
			if (vector[index] < min) {
				min = vector[index];
				tmp = index;
			}
		}
		return tmp;
	}

}
