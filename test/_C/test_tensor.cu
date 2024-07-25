//
// Created by JUMPWORK-SERVER-01 on 2024/7/24.
//

#include<iostream>
#include"../../minitorch/_C/include/base.h"

namespace mt = minitorch;


int main() {
    std::cout << "Start Test!" << std::endl;
    auto tensor = zeros(mt::Shape({3, 2, 4}), mt::DType(mt::DTypeEnum::KFloat32), mt::DeviceEnum::KCpu);
    std::vector<mt::Slice> slices = {mt::Slice(size_t(1), size_t(2), size_t(1))};
    auto tensor_b = tensor.get_item(slices);
    return 0;
}
