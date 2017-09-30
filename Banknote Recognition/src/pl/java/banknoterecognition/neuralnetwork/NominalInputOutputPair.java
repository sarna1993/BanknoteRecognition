package pl.java.banknoterecognition.neuralnetwork;

import java.util.List;

import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.KeyPoint;

public class NominalInputOutputPair {
	private List<KeyPoint> descriptorsInput;
	private List<double[]> output;
	
	public NominalInputOutputPair(List<KeyPoint> descriptorsInput, List<double[]> output) {
		this.descriptorsInput = descriptorsInput;
		this.output = output;
	}
	public List<KeyPoint> getDescriptorsInput() {
		return descriptorsInput;
	}
	public void setDescriptorsinput(List<KeyPoint> descriptorsInput) {
		this.descriptorsInput = descriptorsInput;
	}
	public List<double[]> getOutput() {
		return output;
	}
	public void setOutput(List<double[]> output) {
		this.output = output;
	}
	
	
	
}
