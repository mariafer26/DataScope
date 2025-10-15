from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

#  import io UNUSED


class StatsTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_stats_render_for_numeric_csv(self):
        csv_content = b"city,sales\nA,10\nB,20\nC,30\n"
        file = SimpleUploadedFile("sales.csv", csv_content, content_type="text/csv")
        resp = self.client.post(reverse("upload_file"), {"file": file})
        self.assertContains(resp, "Basic metrics", status_code=200)
        self.assertContains(resp, "sales")
        self.assertContains(resp, "Mean")
