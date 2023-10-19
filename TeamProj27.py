import matplotlib.pyplot as plt
import numpy as np
#from get_coordinates_in_image import onclick
from get_coordinates_in_image import get_click_coordinates


#masking is highlighting a specific portion of the image and setting the rest of image to 0 
#array slicing is like cropping, creating a new smaller array with only the cropped part

def load_image(filename):
    image = plt.imread(filename)
    return image

def grayscale(img):
    grayscale_vector = np.array([0.2989, 0.5870, 0.1140])
    grayscale_image = np.dot(img, grayscale_vector)
    return grayscale_image

def createGaussian():
    gaussian_filter = np.array([[1, 4, 6, 4, 1],
                                [4, 16, 24, 16, 4],
                                [6, 24, 36, 24, 6],
                                [4, 16, 24, 16, 4],
                                [1, 4, 6, 4, 1]]) / 256
    return gaussian_filter

def gaussian(img, filter):

    filtered = np.zeros_like(img)
    kernelSize = filter.shape[0]
    padSize = kernelSize // 2

    for i in range(padSize, img.shape[0] - padSize):
        for j in range(padSize, img.shape[1] - padSize):
            window = img[i - padSize:i + padSize + 1, j - padSize:j + padSize + 1]
            filtered[i, j] = int(np.sum(window * filter))

    return filtered

    # filtered_image = np.zeros_like(img)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         filtered_image[i, j] = np.sum(img[i-2:i+3, j-2:j+3] * filter)
    # return filtered_image

def subsample(img):
    # something about can't divide by odd numnbers
    # subsampleimg = img[::2, ::2]
    # return subsampleimg

    rows, cols = img.shape
    subsampled_rows = (rows + 1) // 2
    subsampled_cols = (cols + 1) // 2
    subsampleimg = np.zeros((subsampled_rows, subsampled_cols))

    for i in range(subsampled_rows):
        for j in range(subsampled_cols):
            subsampleimg[i, j] = img[2 * i, 2 * j]

    return subsampleimg

def pad_or_crop(image, target_shape, sourceCoords, targetCoords):
    original_shape = image.shape
    x1 = sourceCoords[0]
    y1 = sourceCoords[1]
    x2 = targetCoords[0]
    y2 = targetCoords[1]
    diff_x = x2 - x1
    diff_y = y2 - y1

    if diff_x > 0:  # pad x
        pad_left = diff_x
        pad_right = target_shape[1]-len(image[0])-diff_x
        image = np.pad(image, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
    elif diff_x < 0:  # crop x
        crop_left = -diff_x // 2
        crop_right = original_shape[1] + diff_x - crop_left
        image = image[:, crop_left:crop_right]

    if diff_y > 0:  # pad y
        pad_top = diff_y
        pad_bottom = target_shape[0]-len(image)-diff_y
        image = np.pad(image, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
    elif diff_y < 0:  # crop y
        crop_top = -diff_y // 2
        crop_bottom = original_shape[0] + diff_y - crop_top
        image = image[crop_top:crop_bottom, :]

    return image

def upsample(image):
    h = np.zeros((len(image)*2, len(image[0])*2))
    for i in range(len(h)):
        for j in range(len(h[i])):
            if i%2==0 and j%2==0:
                h[i][j] = image[int(i/2)][int(j/2)]
    #print("Step 1")
    #print(h)
    for i in range(1, len(h)-1):
        for j in range(1, len(h[i])-1):
            if h[i-1][j-1]!=0 and h[i+1][j-1]!=0 and h[i-1][j+1]!=0 and h[i+1][j+1]!=0:
                x1 = (h[i-1][j-1] + h[i+1][j-1])/2
                x2 = (h[i-1][j+1] + h[i+1][j+1])/2
                avg = (x1+x2)/2
                h[i][j] = avg
            if j+1==len(h[0])-1 and i%2!=0:
                h[i][j+1] = (h[i-1][j] + h[i+1][j])/2
            if i+1==len(h)-1 and j%2!=0:
                h[i+1][j] = (h[i][j-1] + h[i][j+1])/2
    #print("Step 2")
    #print(h)
    for i in range(1, len(h)-1):
        for j in range(1, len(h[i])-1):
            if h[i][j-1]!=0 and h[i][j+1]!=0 and h[i-1][j]!=0 and h[i+1][j]!=0:
                x1 = (h[i][j-1] + h[i][j+1])/2
                x2 = (h[i-1][j] + h[i+1][j])/2
                avg = (x1+x2)/2
                h[i][j] = avg
    #print("Step 3")
    #print(h)
    for i in range(0, len(h)):
        for j in range(0, len(h[i])):
            if i==0 and h[i][j]==0 and j<len(h[i])-1: #skips corners
                h[i][j] = (h[i][j-1] + h[i][j+1] + h[i+1][j])/3
            elif i==len(h)-1 and h[i][j]==0 and j!=0 and j<len(h[i])-1: #skips corners
                if h[i][j+1]==0:
                    h[i][j] = (h[i][j-1] + h[i-1][j])/2
                else:
                    h[i][j] = (h[i][j-1] + h[i][j+1] + h[i-1][j])/3
            elif j==0 and h[i][j]==0 and i<len(h)-1:
                h[i][j] = (h[i-1][j] + h[i+1][j] + h[i][j+1])/3
            elif j==len(h[0])-1 and h[i][j]==0 and i!=0 and i<len(h)-1:
                if h[i+1][j]==0:
                    h[i][j] = (h[i-1][j] + h[i][j-1])/2
                else:
                    h[i][j] = (h[i-1][j] + h[i+1][j] + h[i][j-1])/3
    #print("Step 4")
    #print(h)
    for i in [0, len(h)-1]:
        for j in [0, len(h[i])-1]:
            if h[i][j]==0 and i==len(h)-1 and j==0:
                h[i][j] = (h[i-1][j] + h[i][j+1])/2
            if h[i][j]==0 and i==0 and j==len(h[i])-1:
                h[i][j] = (h[i][j-1] + h[i+1][j])/2
            if h[i][j]==0 and i==len(h)-1 and j==len(h[i])-1:
                h[i][j] = (h[i-1][j] + h[i][j-1])/2
    #print("Step 5")
    #print(h)
    return h

def mask0(img, cutCoords, cropSize):
    # ones for crop, zeros for the bg
    mask = np.zeros_like(img)
    x = cutCoords[0]
    y = cutCoords[1]
    print("coords source: ", cutCoords)
    mask[y:y+cropSize, x:x+cropSize] = 1
    return mask

def mask1(img, cutCoords, cropSize):
    #zeros for crop, ones for the bg
    mask = np.ones_like(img)
    x = cutCoords[0]
    y = cutCoords[1]
    print("coords target: ", cutCoords)
    mask[y:y+cropSize, x:x+cropSize] = 0
    return mask

def subGaussian(img, filter):
    # apply gaussian, subtract from original img
    lap = gaussian(img, filter)
    lap = subsample(lap)
    return lap

def upGaussian(img, filter):
    lap = upsample(img)
    lap = gaussian(img, filter)
    return lap

def pyramid(img, lap, filter):
    next = upsample(img)
    next = gaussian(next[0:len(lap), 0:len(lap[0])], filter[0:len(lap), 0:len(lap[0])])
    next += lap
    return next

def combine_images(source, target, mask0):
    #delta = source - target
    #If source size is smaller than your target size, pad
    #To pad you check if x diff is positive or negative 
    # if it is positive you add x diff number of rows to the left side and
    #  target x length - source x length - x diff to the right side
    #if target > source, cut
    source_height, source_width = source.shape
    target_height, target_width = target.shape
    crop_size = mask0.shape[0]

    if crop_size > target_height or crop_size > target_width: # checks if mask dimensions are larger than the target
        raise ValueError("Error: Crop window is larger than the target image size.")

    if crop_size > source_height or crop_size > source_width: # checks if crop window is larger than the source image size
        raise ValueError("Error: Crop window is larger than the source image size.")

    combined = source + target
    return combined

def main():
    global x_click, y_click
    x_click, y_click = 0, 0  # Initialize click coordinates

    # Get user inputs for filenames, cut size, and pyramid levels
    sourceFile = input("Please input the color source image file: ")
    if (sourceFile == "schrute"):
        sourceFile = "/Users/robertzhang/Downloads/Team Proj ENGR 133/schrute.jpg"
    elif (sourceFile == "scott"):
        sourceFile = "/Users/robertzhang/Downloads/Team Proj ENGR 133/scott.jpg"
    elif (sourceFile == "superman"):
        sourceFile = "/Users/robertzhang/Downloads/Team Proj ENGR 133/superman.jpg"
    elif (sourceFile == "clark kent"):
        sourceFile = "/Users/robertzhang/Downloads/Team Proj ENGR 133/clark kent.jpg"
    elif (sourceFile == "penguin"):
        sourceFile = "/Users/robertzhang/Downloads/Team Proj ENGR 133/penguin.jpg"
    elif (sourceFile == "tiger"):
        sourceFile = "/Users/robertzhang/Downloads/Team Proj ENGR 133/tigerwoods.jpg"
    elif (sourceFile == "brady"):
        sourceFile = "/Users/robertzhang/Downloads/Team Proj ENGR 133/tombrady.jpg"
    elif (sourceFile == "washer"):
        sourceFile = "/Users/robertzhang/Downloads/Team Proj ENGR 133/washer.jpg"
    else:
        sourceFile = "/Users/robertzhang/Downloads/Team Proj ENGR 133/superman.jpg"

    targetFile = input("Please input the color target image file: ")
    if (targetFile == "schrute"):
        targetFile = "/Users/robertzhang/Downloads/Team Proj ENGR 133/schrute.jpg"
    elif (targetFile == "scott"):
        targetFile = "/Users/robertzhang/Downloads/Team Proj ENGR 133/scott.jpg"
    elif (targetFile == "superman"):
        targetFile = "/Users/robertzhang/Downloads/Team Proj ENGR 133/superman.jpg"
    elif (targetFile == "clark kent"):
        targetFile = "/Users/robertzhang/Downloads/Team Proj ENGR 133/clark kent.jpg"
    elif (targetFile == "penguin"):
        targetFile = "/Users/robertzhang/Downloads/Team Proj ENGR 133/penguin.jpg"
    elif (targetFile == "tiger"):
        targetFile = "/Users/robertzhang/Downloads/Team Proj ENGR 133/tigerwoods.jpg"
    elif (targetFile == "brady"):
        targetFile = "/Users/robertzhang/Downloads/Team Proj ENGR 133/tombrady.jpg"
    elif (targetFile == "wall"):
        targetFile = "/Users/robertzhang/Downloads/Team Proj ENGR 133/wall.jpg"
    else:
        targetFile = "/Users/robertzhang/Downloads/Team Proj ENGR 133/superman.jpg"
    cropSize = int(input("Please input the size of the crop window: "))
    num_pyramids = int(input("Please input the number of pyramid levels: "))

    # Load source and target images

    source = load_image(sourceFile)
    source = grayscale(source)
    target = load_image(targetFile)
    target = grayscale(target)

    print("source size: ", np.shape(source))

    # Get cut location from user input
    x1, y1 = get_click_coordinates(source)
    sourceMaskCoords = [x1, y1]
    x2, y2 = get_click_coordinates(target)
    targetMaskCoords = [x2, y2]
    print("coords source: ", sourceMaskCoords)
    print("coords target: ", targetMaskCoords)
    
    xdelta = x1 - x2
    ydelta = y1 - y2

    # Apply Gaussian filter, Laplacian
    gaussian_filter = createGaussian()
    lapArray = []
    sourceLoop = source
    originalSource = source
    targetLoop = target
    originalTarget = target
    combinedLapSource = source # placeholder for combined lap source image
    for i in range(num_pyramids-1):
        sourceMask = mask0(sourceLoop, sourceMaskCoords, cropSize)
        sourceMaskCoords[0] /= 2
        sourceMaskCoords[0] = int(sourceMaskCoords[0])
        sourceMaskCoords[1] /= 2
        sourceMaskCoords[1] = int(sourceMaskCoords[1])
        targetMask = mask1(targetLoop, targetMaskCoords, cropSize)
        targetMaskCoords[0] /= 2
        targetMaskCoords[0] = int(targetMaskCoords[0])
        targetMaskCoords[1] /= 2
        targetMaskCoords[1] = int(targetMaskCoords[1])
        print("sourceMask size: ", np.shape(sourceMask))

        #updates the "source image"
        img1 = subGaussian(sourceLoop, gaussian_filter)
        img2 = subGaussian(targetLoop, gaussian_filter)

        #updates the "source image"
        originalSource = img1
        originalTarget = img2
        
        img1 = upGaussian(img1, gaussian_filter)
        img2 = upGaussian(img2, gaussian_filter)

        if (i == num_pyramids-2):
            img3 = sourceMask * img1[0:len(sourceMask), 0:len(sourceMask[0])]
            img4 = targetMask * img2[0:len(targetMask), 0:len(targetMask[0])]
            img3 = pad_or_crop(img3,img4.shape, sourceMaskCoords, targetMaskCoords)
            combinedLapSource = img3 + img4
            plt.imshow(combinedLapSource, cmap='gray')
            plt.title("combinedimage")
            plt.axis('off')
            plt.show()
         
        # Ensure both source and img1 have the same dimensions
        #img1 = pad_or_crop(img1, img2.shape)
        print("sourceLoop size: ", np.shape(sourceLoop))
        if (i == 0):
            img1 = source - img1[0:len(sourceLoop), 0:len(sourceLoop[0])]
            img2 = target - img2[0:len(targetLoop), 0:len(targetLoop[0])]
        else:
            img1 = originalSource - img1[0:len(sourceLoop), 0:len(sourceLoop[0])]
            img2 = originalTarget - img2[0:len(targetLoop), 0:len(targetLoop[0])]
        
        img1 *= sourceMask
        img2 *= targetMask
        
        # plt.subplot(1, 2, 1)
        # plt.imshow(img1, cmap='gray')
        # plt.title('Source Laplacian')
        # plt.axis('off')
        # plt.subplot(1, 2, 2)
        # plt.imshow(img2, cmap='gray')
        # plt.title('Target Laplacian')
        # plt.axis('off')
        # plt.show()

        img1 = pad_or_crop(img1, img2.shape, sourceMaskCoords, targetMaskCoords)
        lap = combine_images(img1, img2, sourceMask)

        plt.imshow(combinedLapSource, cmap='gray')
        plt.title(f"Laplacian Image - Level {i}")
        plt.axis('off')
        plt.show()

        lapArray.append(lap)

    base = combinedLapSource + lapArray[-1]
    lapArray.pop()
    lapArray.reverse()

    plt.imshow(base, cmap='gray')
    plt.axis('off')
    plt.show()

    for j in range(len(lapArray)):
        base = pyramid(base, lapArray[j], gaussian_filter)
        plt.imshow(base, cmap='gray')
        plt.axis('off')
        plt.show()
    
    
    plt.imshow(base, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
