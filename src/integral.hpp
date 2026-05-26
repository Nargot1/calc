#pragma once

class TIntegral : public IOperation {
public:
    TIntegral(int value)
        : _value(value)
    {}
    int getValue() const{
        return _value;
    }

    virtual ~TIntegral() {}

    virtual void accept(IVisitor& v) {
        v.visitLiteral(*this);
    }

private:
    int _value;
};
