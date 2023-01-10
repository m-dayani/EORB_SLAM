//
// Created by root on 1/11/21.
//

#include "MyRandGen.h"

using namespace std;

namespace EORB_SLAM {

    int MyRandGen::getSimpleInt(int first, int last) {

        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist6(first, last); // distribution in range [1, 6]

        return dist6(rng);
    }

} // EORB_SLAM