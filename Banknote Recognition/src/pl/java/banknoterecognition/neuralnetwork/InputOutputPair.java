package pl.java.banknoterecognition.neuralnetwork;

public class InputOutputPair {
	private double[] descriptorInput;
	private double[] output;
	
	public InputOutputPair(double[] descriptorsInput, double[] output) {
		this.descriptorInput = descriptorsInput;
		this.output = output;
	}

	public double[] getDescriptorInput() {
		return descriptorInput;
	}

	public void setDescriptorInput(double[] descriptorInput) {
		this.descriptorInput = descriptorInput;
	}

	public double[] getOutput() {
		return output;
	}

	public void setOutput(double[] output) {
		this.output = output;
	}
	
}
