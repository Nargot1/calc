#pragma once
class TAddOperation;
class TIntegral;
class MulOperation;

class IVisitor {
public:
    virtual void visitAdd(const TAddOperation& op) = 0;
    virtual void visitMul(const MulOperation& op) = 0;
    virtual void visitLiteral(const TIntegral& op) = 0;
protected:
    virtual ~IVisitor() {}
};

