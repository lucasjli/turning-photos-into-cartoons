package com.celanim.cartoonify;

import org.jocl.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import javax.imageio.ImageIO;

import static org.jocl.CL.*;

/**
 * Processes lots of photos and uses edge detection and colour reduction to make them cartoon-like.
 * <p>
 * Run <code>main</code> to see the usage message.
 * Each input image, eg. xyz.jpg, is processed and then output to a file called xyz_cartoon.jpg.
 * <p>
 * Implementation Note: this class maintains a stack of images, with the original image being
 * at the bottom of the stack (position 0), and the current image being at the top
 * of the stack (this can be accessed as index -1).  Image processing methods
 * should create a new image (1D array of int pixels in row-major order) and push
 * it on top of the stack.  They should not modify images destructively.
 *
 * @author Mark.Utting
 */
public class Cartoonify {

    /**
     * The number of bits used for each colour channel.
     */
    public static final int COLOUR_BITS = 8;

    /**
     * Each colour channel contains a colour value from 0 up to COLOUR_MASK (inclusive).
     */
    public static final int COLOUR_MASK = (1 << COLOUR_BITS) - 1; // eg. 0xFF

    /**
     * An all-black pixel.
     */
    public final int black = createPixel(0, 0, 0);

    /**
     * An all-white pixel.
     */
    public final int white = createPixel(COLOUR_MASK, COLOUR_MASK, COLOUR_MASK);

    // colours are packed into an int as four 8-bit fields: (0, red, green, blue).
    /**
     * The number of the red channel.
     */
    public static final int RED = 2;

    /**
     * The number of the green channel.
     */
    public static final int GREEN = 1;

    /**
     * The number of the blue channel.
     */
    public static final int BLUE = 0;

    /**
     * What level of colour change should be considered an edge.
     */
    private int edgeThreshold = 128;

    /**
     * Number of values in each colour channel (R, G, B) after quantization.
     */
    private int numColours = 3;

    private boolean debug = false;

    private boolean useGPU = false;

    /**
     * The width of all the images.
     */
    private int width;

    /**
     * The height of all the images.
     */
    private int height;

    /**
     * A stack of images, with the current one at position <code>currImage</code>.
     */
    private int[][] pixels;

    /**
     * The position of the current image in the pixels array. -1 means no current image.
     */
    private int currImage;

    /**
     * Create a new photo-to-cartoon processor.
     * <p>
     * The initial stack of images will be empty, so <code>loadPhoto(...)</code>
     * should typically be the first method called.
     */
    public Cartoonify() {
        pixels = new int[4][];
        currImage = -1;  // no image loaded initially
    }

    /**
     * @return What level of colour change should be considered an edge.
     */
    public int getEdgeThreshold() {
        return edgeThreshold;
    }

    /**
     * Set the level of colour change that should be considered an edge.
     * Small numbers (e.g. 50) give lots of heavy black edges.
     * Large numbers (e.g. 1000) give fewer, thinner edges.
     *
     * @param edgeThreshold from 0 up to 1000 or more.
     */
    public void setEdgeThreshold(int edgeThreshold) {
        if (edgeThreshold < 0) {
            throw new IllegalArgumentException("edge threshold must be at least zero, not " + edgeThreshold);
        }
        this.edgeThreshold = edgeThreshold;
    }

    /**
     * @return Number of values in each colour channel (R, G, B) after quantization.
     */
    public int getNumColours() {
        return numColours;
    }

    /**
     * Set the number of values in each colour channel (R, G, B) after quantization.
     */
    public void setNumColours(int numColours) {
        if (0 < numColours && numColours <= 256) {
            this.numColours = numColours;
        } else {
            throw new IllegalArgumentException("NumColours must be 0..256, not " + numColours);
        }
    }

    public boolean isDebug() {
        return debug;
    }

    /**
     * Set this to true to print out extra timing information and save the intermediate photos.
     *
     * @param debug
     */
    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    /**
     * @return the number of images currently on the stack of images.
     */
    public int numImages() {
        return currImage + 1;
    }

    /**
     * Returns an internal representation of all the pixels.
     *
     * @return all the pixels in the current image that is on top of the stack.
     */
    protected int[] currentImage() {
        return pixels[currImage];
    }

    /**
     * Push the given image onto the stack of images.
     *
     * @param newPixels must be the same size (width * height pixels), and contain RGB pixels.
     */
    protected void pushImage(int[] newPixels) {
        assert newPixels.length == width * height;
        currImage++;
        if (currImage >= pixels.length) {
            // expand the maximum number of possible images.
            pixels = Arrays.copyOf(pixels, pixels.length * 2);
        }
        pixels[currImage] = newPixels;
    }

    /**
     * Remove the current image off the stack.
     *
     * @return all the pixels in that image.
     */
    protected int[] popImage() {
        final int[] result = pixels[currImage];
        pixels[currImage--] = null;
        return result;
    }

    /**
     * Push a shallow copy of the given image onto the stack.
     * For speed, this copies the pointer to the image, but does not
     * duplicate all the pixels in the image.
     * <p>
     * Negative numbers are relative to the top of the stack, so -1 means duplicate
     * the current top of the stack.  Zero or positive is relative to the bottom of
     * the stack, so 0 means duplicate the original photo.
     *
     * @param which the number of the photo to duplicate. From <code>-numImages() .. numImages()-1</code>.
     */
    public void cloneImage(int which) {
        final int stackPos = which >= 0 ? which : (currImage + which + 1);
        assert 0 <= stackPos && stackPos <= currImage;
        pushImage(Arrays.copyOf(pixels[stackPos], width * height));
    }

    /**
     * Reset the stack of images so that it is empty.
     */
    public void clear() {
        Arrays.fill(pixels, null);
        currImage = -1;
    }

    /**
     * Loads a photo from the given file.
     * <p>
     * If the stack of photos is empty, this also sets the width and height of
     * images being processed, otherwise it checks that the new image is the
     * same size as the current images.
     *
     * @param filename
     * @throws IOException if the image cannot be read or is the wrong size.
     */
    public void loadPhoto(String filename) throws IOException {
        BufferedImage image = ImageIO.read(new File(filename));
        if (image == null) {
            throw new RuntimeException("Invalid image file: " + filename);
        }
        if (numImages() == 0) {
            width = image.getWidth();
            height = image.getHeight();
        } else if (width != image.getWidth() || height != image.getHeight()) {
            throw new IOException("Incorrect image size: " + filename);
        }
        int[] newPixels = image.getRGB(0, 0, width, height, null, 0, width);
        for (int i = 0; i < newPixels.length; i++) {
            newPixels[i] &= 0x00FFFFFF; // remove any alpha channel, since we will use RGB only
        }
        pushImage(newPixels);
    }

    /**
     * Save the current photo to disk with the given filename.
     * <p>
     * Does not change the stack of images.
     *
     * @param newName the extension of this name (eg. .jpg) determines the output file type.
     * @throws IOException
     */
    public void savePhoto(String newName) throws IOException {
        BufferedImage image = new BufferedImage(width, height,
                BufferedImage.TYPE_INT_RGB);
        image.setRGB(0, 0, width, height, currentImage(), 0, width);
        final int dot = newName.lastIndexOf('.');
        final String extn = newName.substring(dot + 1);
        final File outFile = new File(newName);
        ImageIO.write(image, extn, outFile);
    }

    /**
     * @return the width of the current images that we are processing.
     */
    public int width() {
        return width;
    }

    /**
     * @return the height of the current images that we are processing.
     */
    public int height() {
        return height;
    }

    /**
     * Adds a new image that is a grayscale version of the current image.
     */
    public void grayscale() {
        int[] oldPixels = currentImage();
        int[] newPixels = new int[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = oldPixels[y * width + x];
                int average = (red(rgb) + green(rgb) + blue(rgb)) / 3;
                int newRGB = createPixel(average, average, average);
                newPixels[y * width + x] = newRGB;
            }
        }
        pushImage(newPixels);
    }

    public static final int[] GAUSSIAN_FILTER = {
            2, 4, 5, 4, 2, // sum=17
            4, 9, 12, 9, 4, // sum=38
            5, 12, 15, 12, 5, // sum=49
            4, 9, 12, 9, 4, // sum=38
            2, 4, 5, 4, 2  // sum=17
    };
    public static final double GAUSSIAN_SUM = 159.0;

    /**
     * Adds one new image that is a blurred version of the current image.
     */
    public void gaussianBlur() {
        long startBlur = System.currentTimeMillis();
        int[] newPixels = new int[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int red = clamp(convolution(x, y, GAUSSIAN_FILTER, RED) / GAUSSIAN_SUM);
                int green = clamp(convolution(x, y, GAUSSIAN_FILTER, GREEN) / GAUSSIAN_SUM);
                int blue = clamp(convolution(x, y, GAUSSIAN_FILTER, BLUE) / GAUSSIAN_SUM);
                newPixels[y * width + x] = createPixel(red, green, blue);
            }
        }
        pushImage(newPixels);
        long endBlur = System.currentTimeMillis();
        if (debug) {
            System.out.println("  gaussian blurring took " + (endBlur - startBlur) / 1e3 + " secs.");
        }
    }


    public static final int[] SOBEL_VERTICAL_FILTER = {
            -1, 0, +1,
            -2, 0, +2,
            -1, 0, +1
    };

    public static final int[] SOBEL_HORIZONTAL_FILTER = {
            +1, +2, +1,
            0, 0, 0,
            -1, -2, -1
    };

    /**
     * Detects edges in the current image and adds an image where black pixels
     * mark the edges and the other pixels are all white.
     * <p>
     * The <code>getEdgeThreshold()</code> value determines how aggressive
     * the edge-detection is.  Small values (e.g. 50) mean very aggressive,
     * while large values (e.g. 1000) generate few edges.
     */
    public void sobelEdgeDetect() {
        long startEdges = System.currentTimeMillis();
        int[] newPixels = new int[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int redVertical = convolution(x, y, SOBEL_VERTICAL_FILTER, RED);
                int greenVertical = convolution(x, y, SOBEL_VERTICAL_FILTER, GREEN);
                int blueVertical = convolution(x, y, SOBEL_VERTICAL_FILTER, BLUE);
                int redHorizontal = convolution(x, y, SOBEL_HORIZONTAL_FILTER, RED);
                int greenHorizontal = convolution(x, y, SOBEL_HORIZONTAL_FILTER, GREEN);
                int blueHorizontal = convolution(x, y, SOBEL_HORIZONTAL_FILTER, BLUE);
                int verticalGradient = Math.abs(redVertical) + Math.abs(greenVertical) + Math.abs(blueVertical);
                int horizontalGradient = Math.abs(redHorizontal) + Math.abs(greenHorizontal) + Math.abs(blueHorizontal);
                // we could take use sqrt(vertGrad^2 + horizGrad^2), but simple addition catches most edges.
                int totalGradient = verticalGradient + horizontalGradient;
                if (totalGradient >= edgeThreshold) {
                    newPixels[y * width + x] = black; // we colour the edges black
                } else {
                    newPixels[y * width + x] = white;
                }
            }
        }
        pushImage(newPixels);
        long endEdges = System.currentTimeMillis();
        if (debug) {
            System.out.println("  sobel edge detect took " + (endEdges - startEdges) / 1e3 + " secs.");
        }
    }

    /**
     * Adds a new image that is the same as the current image but with fewer colours.
     * <p>
     * The <code>getNumColours()</code> setting determines the desired number of
     * colour values in EACH colour channel after this method finishes.
     */
    public void reduceColours() {
        long startQuantize = System.currentTimeMillis();
        int[] oldPixels = currentImage();
        int[] newPixels = new int[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = oldPixels[y * width + x];
                int newRed = quantizeColour(red(rgb), numColours);
                int newGreen = quantizeColour(green(rgb), numColours);
                int newBlue = quantizeColour(blue(rgb), numColours);
                int newRGB = createPixel(newRed, newGreen, newBlue);
                newPixels[y * width + x] = newRGB;
            }
        }
        pushImage(newPixels);
        long endQuantize = System.currentTimeMillis();
        if (debug) {
            System.out.println("  colour reduction took  " + (endQuantize - startQuantize) / 1e3 + " secs.");
        }
    }

    /**
     * Converts the given colour value (eg. 0..255) to an approximate colour value.
     * This is a helper method for reducing the number of colours in the image.
     * <p>
     * For example, if numPerChannel is 3, then:
     * <ul>
     *   <li>0..85 will be mapped to 0;</li>
     *   <li>86..170 will be mapped to 127;</li>
     *   <li>171..255 will be mapped to 255;</li>
     * </ul>
     * So the output colour values always start at 0, end at COLOUR_MASK, and any other
     * values are spread out evenly in between.  This requires some careful maths, to
     * avoid overflow and to divide the input colours up into <code>numPerChannel</code>
     * equal-sized buckets.
     *
     * @param colourValue   0 .. COLOUR_MASK
     * @param numPerChannel how many colours we want in the output.
     * @return a discrete colour value (0..COLOUR_MASK).
     */
    int quantizeColour(int colourValue, int numPerChannel) {
        float colour = colourValue / (COLOUR_MASK + 1.0f) * numPerChannel;
        //IMPORTANT NOTE: due to the different implemention of the "round" funciton in OpenCL, 
        //you need to use 0.49999f instead of 0.5f in your kernel code
        int discrete = Math.round(colour - 0.5f);
        assert 0 <= discrete && discrete < numPerChannel;
        int newColour = discrete * COLOUR_MASK / (numPerChannel - 1);
        assert 0 <= newColour && newColour <= COLOUR_MASK;
        return newColour;
    }

    /**
     * Merges a mask image on top of another image.
     * <p>
     * Since this operation takes two input images, it allows the caller
     * to specify those images by their position.  The input images are
     * left unchanged, and the new merged image is pushed on top of the stack.
     *
     * @param maskImage  the number/position of the mask (as for cloneImage).
     * @param maskColour an exact pixel colour.  Where the mask is this colour,
     *                   the other image will be chosen.
     * @param otherImage the number/position of the underneath image.
     */
    public void mergeMask(int maskImage, int maskColour, int otherImage) {
        long startMasking = System.currentTimeMillis();
        cloneImage(maskImage);
        int[] maskPixels = popImage();
        cloneImage(otherImage);
        int[] photoPixels = popImage();
        int[] newPixels = new int[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                if (maskPixels[index] == maskColour) {
                    newPixels[index] = photoPixels[index];
                } else {
                    newPixels[index] = maskPixels[index];
                }
            }
        }
        pushImage(newPixels);
        long endMasking = System.currentTimeMillis();
        if (debug) {
            System.out.println("  masking edges took     " + (endMasking - startMasking) / 1e3 + " secs.");
        }
    }

    /**
     * This applies the given N*N filter around the pixel (xCentre,yCentre).
     * <p>
     * Applying a filter means multiplying each nearby pixel (within the N*N box)
     * by the corresponding factor in the filter array (which is conceptually a 2D matrix).
     * <p>
     * This method does not change the current image at all.  It just multiplies
     * the filter matrix by the colour values of the pixels around (xCentre,yCentre)
     * and returns the resulting (integer) value.
     * <p>
     * This method is 'package-private' (the default protection) so that the tests can test it.
     *
     * @param xCentre
     * @param yCentre
     * @param filter  a 2D square matrix, laid out in row-major order in a 1D array.
     * @param colour  which colour to apply the filter to.
     * @return the sum of multiplying the requested colour of each pixel by its filter factor.
     */
    // Optimized version of convolution() function
    int convolution(int xCentre, int yCentre, int[] filter, int colour) {
        int sum = 0;

        // Compute filter size using sqrt instead of iterative computation
        int filterSize = (int) Math.sqrt(filter.length);
        if (filterSize * filterSize != filter.length) {
            throw new IllegalArgumentException("non-square filter: " + Arrays.toString(filter));
        }
        final int filterHalf = filterSize >> 1; // Use bit shift instead of division

        // Precompute wrapped coordinates to avoid redundant calculations
        int[] wrappedY = new int[filterSize];
        int[] wrappedX = new int[filterSize];
        for (int i = 0; i < filterSize; i++) {
            wrappedY[i] = wrap(yCentre + i - filterHalf, height) * width; // Precompute y * width
            wrappedX[i] = wrap(xCentre + i - filterHalf, width);
        }

        // Cache the current image to avoid redundant function calls
        int[] image = pixels[currImage];

        // Perform convolution
        for (int filterY = 0; filterY < filterSize; filterY++) {
            int yOffset = wrappedY[filterY]; // Use precomputed y offset
            for (int filterX = 0; filterX < filterSize; filterX++) {
                int x = wrappedX[filterX];
                int rgb = image[yOffset + x]; // Directly access pixel array, avoiding function call

                int filterVal = filter[filterY * filterSize + filterX];

                // Inline colourValue computation to avoid function call
                sum += ((rgb >> (colour * COLOUR_BITS)) & COLOUR_MASK) * filterVal;
            }
        }
        // System.out.println("convolution(" + xCentre + ", " + yCentre + ") = " + sum);
        return sum;
    }

    /**
     * Restricts an index to be within the image.
     * <p>
     * Different strategies are possible for this, such as wrapping around,
     * clamping to 0 and size-1, or reflecting off the edge.
     * <p>
     * The current implementation reflects off each edge.
     *
     * @param pos  an index that might be slightly outside the image boundaries.
     * @param size the width of the image (for x value) or the height (for y values).
     * @return the new index, which is in the range <code>0 .. size-1</code>.
     */
    // Optimized version wrap() function
    public int wrap(int pos, int size) {
        /*if (pos < 0) {
            pos = -1 - pos;
        } else if (pos >= size) {
            pos = (size - 1) - (pos - size);
        }
        assert 0 <= pos;
        assert pos < size;
        return pos;*/
    // Avoid branching (if-else) and improves performance by reducing CPU pipeline stalls
    return Math.max(0,Math.min(pos,size-1));
    }

    /**
     * Clamp a colour value to be within the allowable range for each colour.
     *
     * @param value a floating point colour value, which may be out of range.
     * @return an integer colour value, in the range <code>0 .. COLOUR_MASK</code>.
     */
    public int clamp(double value) {
        int result = (int) (value + 0.5); // round to nearest integer
        if (result <= 0) {
            return 0;
        } else if (result > COLOUR_MASK) {
            return 255;
        } else {
            return result;
        }
    }

    /**
     * Get a pixel from within the current photo.
     *
     * @param x must be in the range <code>0 .. width-1</code>.
     * @param y must be in the range <code>0 .. height-1</code>.
     * @return the requested pixel of the current image, in RGB format.
     * @throws ArrayIndexOutOfBoundsException exception if there is no current image.
     */
    public int pixel(int x, int y) {
        return currentImage()[y * width + x];
    }

    /**
     * Extract a given colour channel out of the given pixel.
     *
     * @param pixel  an RGB value.
     * @param colour one of RED, GREEN or BLUE.
     * @return a colour value, ranging from 0 .. COLOUR_MASK.
     */
    public final int colourValue(int pixel, int colour) {
        return (pixel >> (colour * COLOUR_BITS)) & COLOUR_MASK;
    }

    /**
     * Get the red value of the given pixel.
     *
     * @param pixel an RGB value.
     * @return a value in the range 0 .. COLOUR_MASK
     */
    public final int red(int pixel) {
        return colourValue(pixel, RED);
    }

    /**
     * Get the green value of the given pixel.
     *
     * @param pixel an RGB value.
     * @return a value in the range 0 .. COLOUR_MASK
     */
    public final int green(int pixel) {
        return colourValue(pixel, GREEN);
    }

    /**
     * Get the blue value of the given pixel.
     *
     * @param pixel an RGB value.
     * @return a value in the range 0 .. COLOUR_MASK
     */
    public final int blue(int pixel) {
        return colourValue(pixel, BLUE);
    }

    /**
     * Constructs one integer RGB pixel from the individual components.
     *
     * @param redValue
     * @param greenValue
     * @param blueValue
     * @return
     */
    public final int createPixel(int redValue, int greenValue, int blueValue) {
        assert 0 <= redValue && redValue <= COLOUR_MASK;
        assert 0 <= greenValue && greenValue <= COLOUR_MASK;
        assert 0 <= blueValue && blueValue <= COLOUR_MASK;
        return (redValue << (2 * COLOUR_BITS)) + (greenValue << COLOUR_BITS) + blueValue;
    }

    /**
     * Processes one input photo, applying all the desired transformations to it.
     * Saves the resulting photo in a new file of the same type.
     * E.g. if the input file is "foo.jpg" the output file will be "foo_cartoon.jpg".
     *
     * @param name path to the photo, including a known extension (e.g. ".jpg").
     * @return the number of milliseconds to process this photo (excluding loading/saving).
     * @throws IOException
     */
    protected long processPhoto(String name) throws IOException {
        //no need to change the implementation of this method
        int dot = name.lastIndexOf(".");
        if (dot <= 0) {
            System.err.println("Skipping unknown kind of file: " + name);
            return 0L;
        }
        final String baseName = name.substring(0, dot);
        final String extn = name.substring(dot).toLowerCase();
        loadPhoto(name);
        final String newName = baseName + "_cartoon" + extn;
        //Please do NOT change the start of time measurement
        final long time0 = System.currentTimeMillis();
        if (useGPU) {
            processPhotoOpenCL();
        } else {
            processPhotoOnCPU();
        }
        //Please do NOT change the end of time measurement
        long time1 = System.currentTimeMillis();
        //Please do NOT remove or change this output message
        System.out.println("Done " + name + " -> " + newName + " in " + (time1 - time0) / 1e3 + " secs.");
        savePhoto(newName);
        if (debug) {
            // At this stage the stack of images is (from bottom to top):
            //  original, blurred, edges, original, quantized, final
            popImage();
            savePhoto(baseName + "_colours" + extn);
            popImage();
            popImage();
            savePhoto(baseName + "_edges" + extn);
            popImage();
            savePhoto(baseName + "_blurred" + extn);
            popImage();
            assert numImages() == 1;
        }
        clear();
        return time1 - time0;
    }


    /**
     * Implement this method to process one input photo on GPU or GPU and CPU
     */
    protected void processPhotoOpenCL() {
        // OpenCL objects initialization
        cl_context context = null;
        cl_command_queue queue1 = null, queue2 = null;  // Define two command queues for parallel execution
        cl_program program = null;
        cl_kernel blurKernel = null, sobelKernel = null, reduceKernel = null, mergeKernel = null;
        cl_mem[] buffers = new cl_mem[5]; // 0:input, 1:blurOut, 2:edgeOut, 3:quantOut, 4:finalOut
        cl_event blurEvent = null, sobelEvent = null, reduceEvent = null, mergeEvent = null;

        try {
            // ========== 1. Load and compile OpenCL program ========== //
            final String srcCode = JOCLUtil.readResourceToString("/com/celanim/cartoonify/kernel.cl");
            System.out.println("Reading 'kernel.cl' file completed!");

            // Initialize OpenCL environment
            cl_platform_id platform = getFirstPlatform();
            cl_device_id device = getFirstDevice(platform);
            context = clCreateContext(null, 1, new cl_device_id[]{device}, null, null, null);

            // Create two command queues for parallel execution
            queue1 = clCreateCommandQueue(context, device, 0, null);
            queue2 = clCreateCommandQueue(context, device, 0, null);

            // Prepare image buffers
            int[] currentPixels = currentImage();
            int[] blurredPixels = new int[width * height];
            int[] edgePixels = new int[width * height];
            int[] quantizedPixels = new int[width * height];
            int[] finalPixels = new int[width * height];

            // ========== 2. Create memory buffers ========== //
            buffers[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_int * currentPixels.length, Pointer.to(currentPixels), null);
            buffers[1] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                    Sizeof.cl_int * blurredPixels.length, null, null);
            buffers[2] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                    Sizeof.cl_int * edgePixels.length, null, null);
            buffers[3] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                    Sizeof.cl_int * quantizedPixels.length, null, null);
            buffers[4] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                    Sizeof.cl_int * finalPixels.length, null, null);

            // ========== 3. Build OpenCL program ========== //
            program = clCreateProgramWithSource(context, 1, new String[]{srcCode}, null, null);
            clBuildProgram(program, 0, null, null, null, null);

            // Create kernels
            blurKernel = clCreateKernel(program, "gaussianBlur", null);
            sobelKernel = clCreateKernel(program, "sobelEdgeDetect", null);
            reduceKernel = clCreateKernel(program, "reduceColours", null);
            mergeKernel = clCreateKernel(program, "mergeMask", null);

            // ========== 4. Execute processing pipeline ========== //
            // Step 1: Gaussian Blur
            setKernelArgs(blurKernel, buffers[0], buffers[1], width, height);
            blurEvent = runKernel2D(queue1, blurKernel, width, height);
            //clFinish(queue1);

            // Step 2: Edge Detection (run on queue2 to parallelize)
            setKernelArgs(sobelKernel, buffers[1], buffers[2], width, height, edgeThreshold);
            sobelEvent = runKernel2D(queue2, sobelKernel, width, height, blurEvent);

            // Step 3: Color Quantization (run on queue2 to parallelize)
            setKernelArgs(reduceKernel, buffers[0], buffers[3], width, height, numColours);
            reduceEvent = runKernel2D(queue2, reduceKernel, width, height);
            //clFinish(queue2);

            // Step 4: Mask Merging (edges + quantized colors)
            setKernelArgs(mergeKernel, buffers[2], buffers[3], buffers[4], 0xFFFFFFFF, width);
            mergeEvent = runKernel2D(queue1, mergeKernel, width, height, sobelEvent, reduceEvent);
            //clFinish(queue1);

            // ========== 5. Retrieve final result ========== //
            readBuffer(queue1, buffers[4], finalPixels, mergeEvent);
            pushImage(finalPixels);

        } catch (Exception ex) {
            System.err.println("OpenCL processing failed: " + ex.getMessage());
            ex.printStackTrace();
        } finally {

            // ========== 6. Cleanup resources ========== //
            releaseResources(new cl_kernel[]{blurKernel, sobelKernel, reduceKernel, mergeKernel}, buffers);
            if (queue1 != null) clReleaseCommandQueue(queue1);
            if (queue2 != null) clReleaseCommandQueue(queue2);
            if (program != null) clReleaseProgram(program);
            if (context != null) clReleaseContext(context);
            if (blurEvent != null) clReleaseEvent(blurEvent);
            if (sobelEvent != null) clReleaseEvent(sobelEvent);
            if (reduceEvent != null) clReleaseEvent(reduceEvent);
            if (mergeEvent != null) clReleaseEvent(mergeEvent);
        }
    }

    // ========== Utility Methods ========== //

    private void setKernelArgs(cl_kernel kernel, Object... args) {
        for (int i = 0; i < args.length; i++) {
            if (args[i] instanceof cl_mem) {
                clSetKernelArg(kernel, i, Sizeof.cl_mem, Pointer.to((cl_mem) args[i]));
            } else if (args[i] instanceof Integer) {
                clSetKernelArg(kernel, i, Sizeof.cl_int, Pointer.to(new int[]{(Integer) args[i]}));
            }
        }
    }

    private cl_event runKernel2D(cl_command_queue queue, cl_kernel kernel, int width, int height, cl_event... waitEvents) {
        cl_event event = new cl_event();
        clEnqueueNDRangeKernel(queue, kernel, 2, null,
                new long[]{width, height}, null, waitEvents == null ? 0 : waitEvents.length, waitEvents.length > 0 ? waitEvents : null, event);
        return event;
    }

    private void readBuffer(cl_command_queue queue, cl_mem buffer, int[] output, cl_event... waitEvents) {
        clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0,
                Sizeof.cl_int * output.length, Pointer.to(output), waitEvents == null ? 0 : waitEvents.length, waitEvents.length > 0 ? waitEvents : null, null);
    }

    private void releaseResources(cl_kernel[] kernels, cl_mem[] buffers) {
        for (cl_kernel kernel : kernels) if (kernel != null) clReleaseKernel(kernel);
        for (cl_mem buffer : buffers) if (buffer != null) clReleaseMemObject(buffer);
    }

    private cl_platform_id getFirstPlatform() {
        cl_platform_id[] platforms = new cl_platform_id[1];
        clGetPlatformIDs(1, platforms, null);
        return platforms[0];
    }

    private cl_device_id getFirstDevice(cl_platform_id platform) {
        cl_device_id[] devices = new cl_device_id[1];
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, devices, null);
        if (devices[0] == null) {
            System.out.println("No GPU found, using CPU");
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, devices, null);
        }
        return devices[0];
    }

    /**
     * Process one input photo step-by-step on CPU
     */
    protected void processPhotoOnCPU() {
        // no need to change the implementation of this method
        // This sequence of processing commands is done to every photo.
        gaussianBlur();
        sobelEdgeDetect();
        int edgeMask = numImages() - 1;
        // now convert the original image into a few discrete colours
        cloneImage(0);
        reduceColours();
        mergeMask(edgeMask, white, -1);
    }


    /**
     * Uses the given command line arguments to set Cartoonify options.
     *
     * @param args     command line arguments
     * @param firstArg the first argument to start at.
     * @return the position of the first non-flag argument.  That is, first file.
     */
    protected int setFlags(String[] args, int firstArg) {
        int currArg = firstArg;
        if ("-g".equals(args[currArg])) {
            useGPU = true;
            currArg += 1;
        }
        if ("-d".equals(args[currArg])) {
            setDebug(true);
            currArg += 1;
        }
        if ("-e".equals(args[currArg])) {
            setEdgeThreshold(Integer.parseInt(args[currArg + 1]));
            System.out.println("Using edge threshold " + getEdgeThreshold());
            currArg += 2;
        }
        if ("-c".equals(args[currArg])) {
            setNumColours(Integer.parseInt(args[currArg + 1]));
            System.out.println("Using " + getNumColours() + " discrete colours per channel.");
            currArg += 2;
        }
        return currArg;
    }

    /**
     * Prints a help/usage message to standard output.
     */
    public void help() {
        System.out.println("Arguments: [-g] [-d] [-e EdgeThreshold] [-c NumColours] photo1.jpg photo2.jpg ...");
        System.out.println("  -g use the GPU, to speed up photo processing.");
        System.out.println("  -d means turn on debugging, which saves intermediate photos.");
        System.out.println("  -e EdgeThreshold values can range from 0 (everything is an edge) up to about 1000 or more.");
        System.out.println("  -c NumColours is the number of discrete values within each colour channel (2..256).");
    }

    /**
     * Run this with no arguments to see the usage message.
     *
     * @param args command line arguments
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        //no need to change the implementation of the main method
        Cartoonify cartoon = new Cartoonify();
        if (args.length == 0) {
            cartoon.help();
            System.exit(1);
        }
        int arg = cartoon.setFlags(args, 0);
        long time = 0;
        int done = 0;
        for (; arg < args.length; arg++) {
            time += cartoon.processPhoto(args[arg]);
            done++;
        }
        //Please do NOT remove or change this output message
        System.out.format("Average processing time is %.3f for %d photos.", time / done / 1e3, done);
    }

}
