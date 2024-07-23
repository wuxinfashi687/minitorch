//
// Created by JUMPWORK-SERVER-01 on 2024/7/23.
//

#ifndef IDENTITY_HPP
#define IDENTITY_HPP


namespace minitorch {
    namespace ops {
        Tensor identity_host_forward(Tensor x);
        Tensor identity_host_backward(Tensor y);
        template<typename T>
        void identity_host_forward_kenerl(T* x);
        template<typename T>
        void identity_host_backward_kenerl(T* y);
    }
}


#endif //IDENTITY_HPP
