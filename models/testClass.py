class Rectangle(object):
	def __init__(self,length,breadth):
		self.length = length
		self.breadth = breadth
	def getArea(self):
		print self.length*self.breadth," is area of rectangle"
class Square(Rectangle):
	def __init__(self,side):
		self.side = side
		Rectangle.__init__(self,side,side)
	def getArea(self):
		Rectangle.getArea(self)
s = Square(4)
r = Rectangle(2,4)
s.getArea()
r.getArea()