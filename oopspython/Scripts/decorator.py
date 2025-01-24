def greet(fx):
    def mfx():
        print("hello")
        fx()
        print("hi there")
    return mfx




@greet
def fx():
    print("Good monring")


fx()















