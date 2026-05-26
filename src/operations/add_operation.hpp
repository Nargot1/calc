#pragma once
#include "operation_interface.hpp"
#include "../visitor.hpp"

class TAddOperation : public ABinaryOperation{
public:
    TAddOperation(IOperation* lhs, IOperation* rhs)
        : ABinaryOperation(lhs, rhs)
    {}
    virtual void accept(IVisitor& v) override {
        return v.visitAdd(*this);
    }
};