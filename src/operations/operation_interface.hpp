#pragma once

class IVisitor;

class IOperation{
public:
    virtual void accept(IVisitor& v) = 0;

protected:
    virtual ~IOperation() {}
};

class ABinaryOperation : public IOperation {
public:
    ABinaryOperation(IOperation* lhs, IOperation* rhs)
        :_lhs(lhs),
        _rhs(rhs)
    {}
public:
    IOperation* _lhs;
    IOperation* _rhs;
};
