#include <iostream>
#include "operations/add_operation.hpp"
#include "operations/mul_operation.hpp"
#include "integral.hpp"
#include "visitor.hpp"

class TEvaluate : public IVisitor {
public:
    virtual ~TEvaluate() {}

    virtual void visitAdd(const TAddOperation& op) override final{
       TEvaluate e1, e2;
       op._lhs->accept(e1);
       op._rhs->accept(e2);

       result = e1.result + e2.result;
    }
    virtual void visitMul(const MulOperation& op) override final{
        TEvaluate e1, e2;
        op._lhs->accept(e1);
        op._rhs->accept(e2);

        result = e1.result * e2.result;
    }

    virtual void visitLiteral(const TIntegral& op) override final{
        result = op.getValue();
    }
public:
    int result;
};

class Concat : public IVisitor {
public:
    virtual ~Concat() = default;

    virtual void visitAdd(const TAddOperation& op) override final{
        Concat c1, c2;
        op._lhs->accept(c1);
        op._rhs->accept(c2);

        result = c1.result + "+" + c2.result;
    }

    virtual void visitMul(const MulOperation& op) override final{
        Concat c1, c2;
        op._lhs->accept(c1);
        op._rhs->accept(c2);

        result = c1.result + "*" + c2.result;
    }

    virtual void visitLiteral(const TIntegral& op) override final{
        result = std::to_string(op.getValue());
    }
public:
    std::string result;
};


int main(){
    TIntegral lhs(1);
    TIntegral rhs(2);
    TAddOperation op(&lhs, &rhs);

    TAddOperation op2(&op, &op);
    
    TEvaluate ev;
    op.accept(ev);

    std::cout << ev.result << std::endl;

    op2.accept(ev);
    std::cout << ev.result << std::endl;

    Concat c;

    op2.accept(c);
    std::cout << c.result << std::endl;/// 1+2+1+2

    MulOperation op3(&lhs, &rhs);
    MulOperation op4(&op3, &op2);

    op3.accept(ev);
    std::cout << ev.result << std::endl;

    op4.accept(ev);
    std::cout << ev.result << std::endl;
    return 0;
}
