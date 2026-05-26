#pragma once

class TAddOperation;
class TIntegral;

class IVisitor {
public:
    virtual void visitAdd(const TAddOperation& op) = 0;
    virtual void visitLiteral(const TIntegral& op) = 0;

protected:
    virtual ~IVisitor() {}
};

