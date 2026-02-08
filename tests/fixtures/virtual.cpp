/* Test file for virtual method detection. */

class Base {
public:
    virtual void on_event(int type) {
        /* Base implementation */
    }

    virtual int compute(int x) = 0;

    void non_virtual_method(void) {
        /* Not virtual */
    }

    virtual ~Base() {}
};

class Derived : public Base {
public:
    void on_event(int type) override {
        /* Derived implementation */
    }

    int compute(int x) override {
        return x * 2;
    }

    void derived_only(void) {
        /* Not virtual, not overriding */
    }

    ~Derived() override {}
};

void use_base(Base* b) {
    b->on_event(42);
    int val = b->compute(10);
}
