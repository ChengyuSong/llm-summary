extern void use(int *p, int n);

int foo(int x, int *p) {
    if (x > 0) {
        p[0] = x * 2;
        use(p, 1);
        if (x > 10) {
            p[1] = x + 1;
            use(p, 2);
            return 2;
        }
        return 1;
    } else {
        p[0] = 0;
        use(p, 0);
        return 0;
    }
}
