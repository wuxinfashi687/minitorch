//
// Created by JUMPWORK-SERVER-01 on 2024/7/23.
//

#ifndef ADD_HPP
#define ADD_HPP


namespace minitorch {
    namespace ops {
        Tensor add_host_forward(Tensor x1, Tensor x2);
        Tensor add_host_backward(Tensor y);
        template<typename T>
        void add_host_forward_kenerl(T* x1, T* x2);
        template<typename T>
        void add_host_backward_kenerl(T* y);
    }
}


#endif //ADD_HPP
