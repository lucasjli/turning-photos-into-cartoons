// Guassian Blur kernel
#define GAUSSIAN_SUM 159.0

int wrap(int pos, int size) {
    /*if (pos < 0) {
        pos = -1 - pos;
    } else if (pos >= size) {
        pos = (size - 1) - (pos - size);
    }
    return pos;*/
return max(0, min(pos, size - 1));  // Implement the same logic as Java code
}

__constant int GAUSSIAN_FILTER[25] = {
    2, 4, 5, 4, 2,
    4, 9, 12, 9, 4,
    5, 12, 15, 12, 5,
    4, 9, 12, 9, 4,
    2, 4, 5, 4, 2
};

int getChannel(int pixel, int channel) {
    return (pixel >> (8 * channel)) & 0xFF;
}

int createPixel(int red, int green, int blue) {
    return (255 << 24) | (red << 16) | (green << 8) | blue;
}

int clampColor(float value) {
    int result = (int)(value + 0.5f);
    return result < 0 ? 0 : (result > 255 ? 255 : result);
}

int convolution(__global const int* pixels, int filterSize, int width, int height, int xCentre, int yCentre, int colour) {
    int sum = 0;
    int weightSum = 0;
    int filterHalf = filterSize / 2;

    for (int filterY = 0; filterY < filterSize; filterY++) {
        int y = wrap(yCentre + filterY - filterHalf, height);
        for (int filterX = 0; filterX < filterSize; filterX++) {
            int x = wrap(xCentre + filterX - filterHalf, width);

            int rgb = pixels[y * width + x];
            int filterVal = GAUSSIAN_FILTER[filterY * filterSize + filterX];

            int colourValue = getChannel(rgb, colour);
            sum += colourValue * filterVal;
            weightSum += filterVal;
        }
    }

    return (sum + weightSum / 2) / weightSum;
}


__kernel void gaussianBlur(__global int *pixels, __global int *newPixels,
                           const int width, const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    int red = convolution(pixels, 5, width, height, x, y, 2);
    int green = convolution(pixels, 5, width, height, x, y, 1);
    int blue = convolution(pixels, 5, width, height, x, y, 0);

    red = clampColor((float)red);
    green = clampColor((float)green);
    blue = clampColor((float)blue);

    newPixels[y * width + x] = createPixel(red, green, blue);
}

// Define Sobel filters as global constants
__constant int SOBEL_VERTICAL_FILTER[] = {-1, 0, 1,
                                          -2, 0, 2,
                                          -1, 0, 1};
__constant int SOBEL_HORIZONTAL_FILTER[] = {1,  2,  1,
                                             0,  0,  0,
                                            -1, -2, -1};

// Channel definitions
#define RED 2  // Based on getChannel implementation (0=blue, 1=green, 2=red)
#define GREEN 1
#define BLUE 0

__kernel void sobelEdgeDetect(__global int *pixels, __global int *newPixels,
                             const int width, const int height, const int edgeThreshold) {
    // Get current pixel coordinates
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;  // Boundary check - return if coordinates are out of bounds

    // Initialize gradient accumulators
    int redV = 0, greenV = 0, blueV = 0;  // Vertical gradients
    int redH = 0, greenH = 0, blueH = 0;  // Horizontal gradients

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int px = wrap(x + kx, width);
            int py = wrap(y + ky, height);
            int pixel = pixels[py * width + px];

            int r = getChannel(pixel, RED);
            int g = getChannel(pixel, GREEN);
            int b = getChannel(pixel, BLUE);

            int kidx = (ky + 1) * 3 + (kx + 1);

            // Accumulate vertical gradients
            redV += r * SOBEL_VERTICAL_FILTER[kidx];
            greenV += g * SOBEL_VERTICAL_FILTER[kidx];
            blueV += b * SOBEL_VERTICAL_FILTER[kidx];

            // Horizontal Sobel
            redH += r * SOBEL_HORIZONTAL_FILTER[kidx];
            greenH += g * SOBEL_HORIZONTAL_FILTER[kidx];
            blueH += b * SOBEL_HORIZONTAL_FILTER[kidx];
        }
    }

    // Calculate absolute gradient magnitudes
    int vertGrad = abs(redV) + abs(greenV) + abs(blueV);  // Vertical edge strength
    int horizGrad = abs(redH) + abs(greenH) + abs(blueH); // Horizontal edge strength
    int totalGrad = vertGrad + horizGrad;

    // Thresholding - black for edges, white for non-edges
    /*newPixels[y * width + x] = (totalGrad >= edgeThreshold) ?
        createPixel(0, 0, 0) :  // Black
        createPixel(255, 255, 255);*/  // White
    // Use select() to remove conditional assignments
    newPixels[y * width + x] = select(createPixel(255, 255, 255), createPixel(0, 0, 0), totalGrad >= edgeThreshold);
}

int quantizeColour(int colourValue, int numPerChannel);
__kernel void reduceColours(__global int *oldPixels, __global int *newPixels,
		                    const int width, const int height, const int numColours) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Boundary check
    if (x >= width || y >= height) return;

    // Pixel index calculation
    int rgb = oldPixels[y * width + x];

    // Extract RGB components
    int a = (rgb >> 24) & 0xFF;
    int red = (rgb >> 16) & 0xFF;
    int green = (rgb >> 8) & 0xFF;
    int blue = rgb & 0xFF;

    // Quantize each component with OpenCL-compatible rounding
    int r = quantizeColour(red, numColours);
    int g = quantizeColour(green, numColours);
    int b = quantizeColour(blue, numColours);

    // Combine back into ARGB format
    newPixels[y * width + x] = (a << 24) | (r << 16) | (g << 8) | b;
}

int quantizeColour(int colourValue, int numPerChannel) {
    const int COLOUR_BITS = 8;
    const int COLOUR_MASK = (1 << COLOUR_BITS) - 1;

    // Normalize to [0, numPerChannel) range
    float colour = (float)colourValue / (COLOUR_MASK + 1.0f) * numPerChannel;

    // Discrete with Java-compatible rounding
    int discrete = (int)floor(colour - 0.5f + 0.49999f);

    // Boundary protection
    discrete = max(0, min(discrete, numPerChannel - 1));

    return (discrete * COLOUR_MASK) / (numPerChannel - 1);
}

__kernel void mergeMask(__global int *maskPixels, __global int *photoPixels, __global int *newPixels,
		                const int maskColour, const int width) {
    // Get 2D thread coordinates (current pixel position)
    int x = get_global_id(0);
    int y = get_global_id(1);

    int idx = y * width + x;  // Calculate 1D buffer index from 2D coordinates

    // If mask pixel matches the specified color, use photo pixel
    // Otherwise, keep the original mask pixel value
    /*if (maskPixels[idx] == maskColour) {
        newPixels[idx] = photoPixels[idx];  // Copy entire pixel (including alpha channel) from source photo
    } else {
        newPixels[idx] = maskPixels[idx];  // Preserve non-mask areas from the original mask
    }*/
    // Use select() to remove conditional assignments
    newPixels[idx] = select(maskPixels[idx], photoPixels[idx], maskPixels[idx] == maskColour);
}

