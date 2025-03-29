// Guassian Blur kernel
__kernel void gaussianBlur(__global int *pixels, __global int *newPixels,
                           const int width, const int height) {

    __const int GAUSSIAN_FILTER[25] = {
        2,  4,  5,  4, 2,
        4,  9, 12,  9, 4,
        5, 12, 15, 12, 5,
        4,  9, 12,  9, 4,
        2,  4,  5,  4, 2
    };
    const float GAUSSIAN_SUM = 159.0f;

    // Get current pixel coordinates
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Boundary check
    if (x >= width || y >= height) return;

    // Initialize accumulators for RGB channels
    float redSum = 0.0f, greenSum = 0.0f, blueSum = 0.0f;

    // Apply 5x5 Gaussian filter
    for (int ky = -2; ky <= 2; ky++) {      // Vertical kernel offset
        for (int kx = -2; kx <= 2; kx++) {  // Horizontal kernel offset
            // Clamp pixel coordinates to image boundaries
            int px = clamp(x + kx, 0, width - 1);
            int py = clamp(y + ky, 0, height - 1);
            // Get source pixel value
            int pixel = pixels[py * width + px];

            // Extract RGB components
            int r = (pixel >> 16) & 0xFF;  // Red channel
            int g = (pixel >> 8) & 0xFF;   // Green channel
            int b = pixel & 0xFF;          // Blue channel

            // Calculate kernel index
            int kidx = (ky + 2) * 5 + (kx + 2);
            int weight = GAUSSIAN_FILTER[kidx];

            // Accumulate weighted color values
            redSum += (float)r * weight;
            greenSum += (float)g * weight;
            blueSum += (float)b * weight;
        }
    }

    // Normalize and round the results
    int red = (int)round(redSum / GAUSSIAN_SUM);
    int green = (int)round(greenSum / GAUSSIAN_SUM);
    int blue = (int)round(blueSum / GAUSSIAN_SUM);

    // Combine channels back into ARGB format and store result
    newPixels[y*width+x] = 0xFF000000 | (clamp(red,0,255)<<16) | (clamp(green,0,255)<<8) | clamp(blue,0,255);
}

__kernel void sobelEdgeDetect(__global int *pixels, __global int *newPixels,
                              const int width, const int height, const int edgeThreshold) {

}


__kernel void reduceColours(__global int *oldPixels, __global int *newPixels,
		                    const int width, const int height, const int numColours) {

}

__kernel void mergeMask(__global int *maskPixels, __global int *photoPixels, __global int *newPixels,
		                const int maskColour, const int width) {

}

