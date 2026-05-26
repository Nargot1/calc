#pragma once
#include "operation_interface.hpp"
#include "../visitor.hpp"

class MulOperation : public ABinaryOperation{
public:
    MulOperation(IOperation* lhs, IOperation* rhs)
        : ABinaryOperation(lhs, rhs)
    {}
    virtual void accept(IVisitor& v) override {
        return v.visitMul(*this);
    }
};
