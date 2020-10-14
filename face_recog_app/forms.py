from django import forms

# class PeopleForm(forms.ModelForm):
   
#     name = forms.CharField(max_length=245, label="Enter Name Of Person In Image")
#     class Meta:
#         model = People
#         fields = ('name',  )


# class ImageForm(forms.ModelForm):

#     image = forms.FileField()    
#     class Meta:
#         model = Image
#         fields = ('image', )


class FileFieldForm(forms.Form):
    file_field = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))

class FileFieldForm_2(forms.Form):
    file_field = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': False}))