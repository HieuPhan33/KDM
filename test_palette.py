from PIL import Image
import numpy as np
def get_palette(n, color_map_by_label):
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3

    # Custom color by label
    for label, colors in color_map_by_label.items():
        palette[label * 3: label * 3 + 3] = colors
    return palette

color_map_by_label = {
        0: [213, 213, 215],
        1: [124, 252, 0],
        2: [155, 118, 83],
        3: [255, 0, 0],
        4: [119, 158, 203]
    }
palette = get_palette(256,color_map_by_label)

hsi_label = {
        0:"metal",
        1: "plant",
        2: "soil",
        3: "creature",
        4: "background"
    }
att = np.asarray([[0,0,1,1,2,3,3,4,4],
                  [0,0,1,1,2,3,3,4,4],
                  [0,0,1,1,2,3,3,4,4],
                  [0,0,1,1,2,3,3,4,4],
                  [0,0,1,1,2,3,3,4,4],
                  [0,0,1,1,2,3,3,4,4],
                  [0,0,1,1,2,3,3,4,4],
                  [0,0,1,1,2,3,3,4,4]],dtype=np.uint8)
att = Image.fromarray(att).resize((1024,1024))
att.putpalette(palette)
att.save('test.png')